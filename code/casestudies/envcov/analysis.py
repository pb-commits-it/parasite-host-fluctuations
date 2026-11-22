#!/usr/bin/env python3
"""Registered envcov analysis. Run python3 -I .../analysis.py --phase fields|theory|simulation|report.

Predeclared checks: old R14 rho within 1e-12, T3 slope within 5e-5 and SE within
5e-5; R12/R12b slopes within 0.001 and pair counts exactly; known FE line within
1e-10; thermal constant-temperature crossing exact to one supplied day; engine
checks use the tolerances preserved in envcov_engine.py. No outcome-driven reruns.
Every phase checkpoints to results and refuses to overwrite its completed output.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0,str(HERE))
ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
import argparse,csv,hashlib,json,math,subprocess,time
import numpy as np
from scipy import stats
import envcov_lib as L
import envcov_stats as S
import envcov_engine as E
RESULTS=HERE/'results'; FIGURES=HERE/'figures'; RAW=ROOT/'data/raw'
SITES=L.R14_INCLUDED

def now():return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def clean(obj):
 if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple,np.ndarray)):return [clean(v) for v in obj]
 if isinstance(obj,np.generic):return clean(obj.item())
 if isinstance(obj,float) and not math.isfinite(obj):return None
 return obj

def save(name,obj):
 p=RESULTS/name;p.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
 print('SAVED',name,flush=True)
def table(name,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with (RESULTS/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 print('SAVED',name,'rows',len(rows),flush=True)
def standardized(a):
 a=np.asarray(a,float);return (a-a.mean())/max(float(a.std(ddof=1)),1e-15)
def pca_fit(mat):
 mat=np.asarray(mat,float);mean=mat.mean(0);sd=mat.std(0,ddof=1)
 pc,loading,expl=L.compute_pca1(mat)
 scale=float((((mat-mean)/sd)@loading).std(ddof=1))
 return pc,dict(mean=mean.tolist(),sd=sd.tolist(),loading=loading.tolist(),scale=scale,explained=expl)
def project(row,fit):return float(((np.array([row[k] for k in L.COV_KEYS])-fit['mean'])/fit['sd'])@fit['loading']/fit['scale'])
def fe_records(rows,covs):
 groups=[]
 for g in sorted({r['plot'] for r in rows}):
  rs=[r for r in rows if r['plot']==g]
  groups.append((np.array([r['x'] for r in rs]),np.array([r['y'] for r in rs]),np.array([[r[c] for c in covs] for r in rs]) if covs else None))
 return groups

def fields():
 out=RESULTS/'field_analysis.json'
 if out.exists():raise RuntimeError('Completed field phase exists; use its saved results.')
 started=now();print('FIELD PHASE',started,flush=True)
 # Synthetic checks precede associations.
 rng=np.random.default_rng(20261114);groups=[]
 for j in range(3):
  x=rng.normal(size=18);z=rng.normal(size=(18,1));y=j+.7*x+.2*z[:,0];groups.append((x,y,z))
 b=S.fe_ols_with_covariates(groups)[0]
 assert abs(b-.7)<1e-10 and abs(b-S.fe_ols_dummies_check(groups))<1e-10
 ln,na=L.maturation_rates(np.full(200,20.))
 assert L._days_to_threshold(ln,0)==math.ceil(1/float(ln[0]))
 assert L._days_to_threshold(na,0)==math.ceil(1/float(na[0]))
 print('CHECK synthetic FE and thermal thresholds PASS',flush=True)
 manifest=json.loads((RAW/'envcov/MANIFEST.json').read_text());verified=[]
 for r in manifest:
  p=RAW/'envcov'/r['path'];actual=hashlib.sha256(p.read_bytes()).hexdigest()
  assert actual==r['sha256'],r['path'];verified.append(r['path'])
 print('CHECK manifest hashes',len(verified),'PASS',flush=True)
 cary=S.load_cary_series(RAW/'cary_lyme');neon=S.load_neon_r14_series(RAW/'neon')
 sigma=np.array([neon['sigma'][s] for s in SITES]);nip=np.array([neon['nip_site'][s] for s in SITES]);ntest=np.array([neon['tested'][s] for s in SITES])
 base=S.spearman_perm_exact6(sigma,nip);assert abs(base[0]-17/35)<1e-12
 print('CHECK R14 unadjusted',base,flush=True)
 dm={};cov={};annual=[];spatial=[];durations={};duration_years=[]
 for path in sorted((RAW/'envcov/daymet').glob('*.csv')):
  label=path.stem;dm[label]=L.read_daymet_daily(path)
  pp=RAW/'envcov/nasa_power'/f'{label}.json';pw=L.read_power_daily(pp) if pp.exists() else None
  cov[label]=L.build_annual_covariates(dm[label],pw)
  for y,r in cov[label].items():annual.append(dict(location=label,year=y,**r))
  meta=L.read_daymet_meta(path)
  spatial.append(dict(location=label,**meta,days=len(dm[label]['year']),year_min=int(dm[label]['year'].min()),year_max=int(dm[label]['year'].max()),pixel_centre_distance_km='not returned by endpoint; containing 1-km cell, nominal <=0.707 km',power_lat=pw['grid_lat'] if pw else '',power_lon=pw['grid_lon'] if pw else ''))
 table('annual_covariates.csv',annual);table('weather_locations.csv',spatial)
 hab=L.extract_neon_habitat_summary(RAW/'neon');site_rows=[]
 for i,s in enumerate(SITES):
  years=neon['test_years'][s]
  row=dict(site=s,sigma=sigma[i],nip=nip[i],tested=int(ntest[i]),positive=neon['positive'][s],test_years=years,r1=neon['r1'][s],mean_index=neon['mean_index'][s],mean_log_trapnights=neon['mean_log_tn'][s],**hab[s])
  for k in cov[s+'_tick'][years[0]]:row[k]=float(np.mean([cov[s+'_tick'][y][k] for y in years]))
  mt=L.read_daymet_meta(RAW/'envcov/daymet'/f'{s}_tower.csv');mp=L.read_daymet_meta(RAW/'envcov/daymet'/f'{s}_tick.csv')
  row['tick_to_tower_km']=L.haversine_km(mp['lat'],mp['lon'],mt['lat'],mt['lon'])
  site_rows.append(row)
 pc,fit_neon=pca_fit([[r[k] for k in L.COV_KEYS] for r in site_rows])
 for r,score in zip(site_rows,pc):r['PC1_env']=float(score)
 partial=[];correlations=[]
 for k in ['unadjusted','PC1_env']+L.COV_KEYS+['P_gwettop_gs','P_rh_gs','P_ts_wmin','forest_plot_frac','elevation_m','mean_index','mean_log_trapnights']:
  z=None if k=='unadjusted' else np.array([r[k] for r in site_rows])
  rr,pneg,ppos=base if z is None else S.freedman_lane_partial6(nip,sigma,z)
  wls=S.logit_wls_6sites(nip,ntest,sigma,z) if z is None or z.std()>1e-12 else {}
  alpha=.05/6 if k in L.COV_KEYS else .05
  verdict='supports' if rr<0 and pneg<alpha else ('contradicts' if rr>0 and ppos<alpha else 'not decided')
  partial.append(dict(covariate=k,r=rr,p_negative=pneg,p_positive=ppos,alpha=alpha,verdict=verdict,**wls))
  if z is not None:correlations.append(dict(covariate=k,pearson_sigma=S.pearson_r(sigma,z),spearman_sigma=S.spearman_rho(sigma,z),pearson_nip=S.pearson_rho(nip,z) if False else S.pearson_r(nip,z),spearman_nip=S.spearman_rho(nip,z)))
 print('NEON primary',partial[1],flush=True)
 table('neon_sites.csv',site_rows);table('neon_adjustments.csv',partial);table('neon_correlations.csv',correlations)
 # Independent dummy-variable verification on real T3 records.
 neon_pairs=[]
 for s in SITES:
  for y,p in neon['yearly_nip'][s].items():
   idx=neon['index'].get((s,y-1),0)
   if idx>0 and p>0:
    row=dict(plot=s,mouse_year=y-1,nymph_year=y,x=math.log(idx),y=math.log(p),PC1_env=project(cov[s+'_tick'][y],fit_neon),effort=math.log(neon['effort'][(s,y-1)]),**cov[s+'_tick'][y])
    neon_pairs.append(row)
 t3=[]
 for k in ['unadjusted','PC1_env']+L.COV_KEYS+['effort']:
  gs=fe_records(neon_pairs,[] if k=='unadjusted' else [k]);b,bz,se,ci,seo,df=S.neon_t3_boot_slope(gs)
  assert abs(b-S.fe_ols_dummies_check(gs))<1e-9
  t3.append(dict(covariate=k,b=b,se=se,ci_low=ci[0],ci_high=ci[1],n=len(neon_pairs)))
 assert len(neon_pairs)==46 and abs(t3[0]['b']-.2464)<5e-5 and abs(t3[0]['se']-.1495)<5e-5
 print('CHECK R14 T3',t3[0],flush=True);table('neon_within_site_slopes.csv',t3)
 # Tower coordinate sensitivity: same six covariates, same primary test.
 tower_mat=[[np.mean([cov[s+'_tower'][y][k] for y in neon['test_years'][s]]) for k in L.COV_KEYS] for s in SITES]
 tower_pc,tower_fit=pca_fit(tower_mat);tower_stat=S.freedman_lane_partial6(nip,sigma,tower_pc)
 # Cary primary PC fitted on collection-year values once, no response used.
 pcc,fit_cary=pca_fit([[cov['CARY'][y][k] for k in L.COV_KEYS] for y in range(1995,2014)])
 for y,row in cov['CARY'].items():row['PC1_env']=project(row,fit_cary)
 cary_sets={};cary_rows=[];cary_corr=[];station_check=[]
 for period in ['primary','holdout','DON_exploratory','LPM_exploratory']:
  rs=[]
  plots=cary['hplots'] if period=='holdout' else cary['dplots']
  for p in plots:
   if period=='holdout':
    for ny,pv in cary['hol_bb'][p].items():
     m=cary['hol_mice_prev'][p].get(ny,0);nn=cary['hol_n'][p].get(ny,0)
     if 2014<=ny<=2022 and nn>=20 and pv>0 and m>0:rs.append(dict(plot=p,mouse_year=ny-1,nymph_year=ny,x=math.log(m),y=math.log(pv),tested=nn))
   else:
    response=cary['dry_nip'] if period=='primary' else cary['dry_don'] if period=='DON_exploratory' else cary['dry_lpm']
    for ny,pv in response[p].items():
     my=ny if period=='LPM_exploratory' else ny-1;m=cary['dry_mice'][p].get(my,0)
     if 1995<=ny<=2013 and (period!='DON_exploratory' or my>=1995) and pv>0 and m>0:rs.append(dict(plot=p,mouse_year=my,nymph_year=ny,x=math.log(m),y=math.log(pv)))
  for r in rs:
   my,ny=r['mouse_year'],r['nymph_year']
   for suffix,y in [('nymph',ny),('larva',my)]:
    for k,v in cov['CARY'][y].items():r[f'{k}_{suffix}']=v
   r['trend']=ny-2004
   r['assay']=cary['dry_pcr_share'].get(r['plot'],{}).get(ny,float('nan'))
   r['chip']=math.log(cary['dry_chip'].get(r['plot'],{}).get(my,0)+1)
   for k,v in cary['station_dry'].get(ny,{}).items():r['station_'+k]=v
  cary_sets[period]=rs;t0=min(r['mouse_year'] for r in rs);t1=max(r['mouse_year'] for r in rs)
  starts=np.random.default_rng(20261114).integers(0,t1-t0,size=(2000,(t1-t0+2)//2))
  adjustments=[('unadjusted',[]),('PC1_nymph',['PC1_env_nymph']),('PC1_larva',['PC1_env_larva'])]
  if period in ['primary','holdout']:
   adjustments += [(k,[k+'_nymph']) for k in L.COV_KEYS]+[('all_six',[k+'_nymph' for k in L.COV_KEYS]),('PC1_trend',['PC1_env_nymph','trend'])]
   if period=='primary':adjustments += [('PC1_assay',['PC1_env_nymph','assay']),('PC1_chip',['PC1_env_nymph','chip'])]+[(k,[k]) for k in ['station_mat','station_degree_days','station_cold_winter_days','station_feb_apr_warm_days','station_snowfall_cm']]
  for label,cols in adjustments:
   rr=[r for r in rs if all(k in r and np.isfinite(r[k]) for k in cols)]
   # Scale each predictor globally; FE slope invariant. Avoid numerical rank loss with DD.
   rr=[dict(r) for r in rr]
   for k in cols:
    values=np.array([r[k] for r in rr]);sd=values.std(ddof=1)
    for r,v in zip(rr,standardized(values)):r[k]=float(v)
   b,bz,se,ci,df=S.cary_block_boot_slope(rr,plots,cols,starts,t0,t1)
   pbar=float(np.mean([v for vals in cary['dry_nip'].values() for v in vals.values()])) if period!='holdout' else float(np.mean([math.exp(r['y']) for r in rs]))
   upper=1-pbar;outside=b+2*se<0 or b-2*se>upper
   verdict='contradicts' if outside else 'not decided'
   cary_rows.append(dict(period=period,covariate=label,b=b,se=se,ci_low=ci[0],ci_high=ci[1],n=len(rr),annual_cohorts=len({r['mouse_year'] for r in rr}),pbar=pbar,upper=upper,shift_gt_registered_1se=abs(b-(.051 if period=='holdout' else .038))>(.118 if period=='holdout' else .058),nonzero_2se=abs(b)>2*se,verdict=verdict,band_compatible=not outside,beta_cov=bz.tolist()))
  print('CARY',period,[r for r in cary_rows if r['period']==period and r['covariate'] in ['unadjusted','PC1_nymph','PC1_larva']],flush=True)
  table('cary_slopes.csv',cary_rows)
  if period=='primary':
   assert len(rs)==102 and abs(cary_rows[0]['b']-.038)<.001
   # Annual aggregates give each calendar cohort one weather value.
   ys=list(range(t0,t1+1));xx=np.array([np.mean([r['x'] for r in rs if r['mouse_year']==y]) for y in ys]);yy=np.array([np.mean([r['y'] for r in rs if r['mouse_year']==y]) for y in ys])
   for lag in ['nymph','larva']:
    for k in L.COV_KEYS+['PC1_env']:
     z=np.array([cov['CARY'][y+(lag=='nymph')][k] for y in ys])
     for target,values in [('log_mice',xx),('log_nip_next',yy)]:
      rr,ci,rho=S.block_boot_corr_ci(values,z,starts)
      cary_corr.append(dict(covariate=k,lag=lag,target=target,r=rr,r_low=ci[0],r_high=ci[1],rho=rho,years=len(ys)))
  if period=='holdout':assert len(rs)==40 and abs([r for r in cary_rows if r['period']==period][0]['b']-.051)<.001
 # Cross-validation uses exactly matched year columns; snow cover != snowfall.
 for sta,dkey in [('mat','mat_cal'),('degree_days','dd0_oct_sep'),('cold_winter_days','cold_winter_frac'),('feb_apr_warm_days','feb_apr_warm_dry_frac'),('snowfall_cm','C4_snow')]:
  ys=sorted(y for y,v in cary['station_dry'].items() if sta in v and y in cov['CARY'])
  a=[cary['station_dry'][y][sta] for y in ys];b=[cov['CARY'][y][dkey] for y in ys]
  station_check.append(dict(station=sta,daymet=dkey,n=len(ys),r=S.pearson_r(a,b),rho=S.spearman_rho(a,b),rmse=float(np.sqrt(np.mean((np.array(a)-b)**2))) if sta!='snowfall_cm' else None))
 table('cary_correlations.csv',cary_corr);table('station_crossvalidation.csv',station_check)
 for s in ['CARY']+SITES:
  lab=s if s=='CARY' else s+'_tick';years=list(range(1994,2013)) if s=='CARY' else [y-1 for y in neon['test_years'][s]]
  variants={}
  for tag,quest,model in [('primary',.02,'wallace2019'),('quest_fast',.04,'wallace2019'),('recalled_powerlaw',.02,'ogden2004_recalled')]:
   d=L.compute_tick_stage_durations(dm[lab],years,quest,model)
   d['n_requested_years']=len(years)
   # NA year-specific outputs, absent from inherited helper's return.
   _,mr=L.maturation_rates(dm[lab]['tmean'],model)
   d['premoult_na_by_year']={y:L._days_to_threshold(mr,int(np.flatnonzero((dm[lab]['year']==y)&(dm[lab]['yday']==152))[0])) for y in years}
   if tag!='recalled_powerlaw':
    assert len(d['phen_ln_by_year'])==len(years) and d['censored_subcohorts']==0
   else:
    print('Recalled sensitivity censoring',s,d['censored_subcohorts'],flush=True)
   variants[tag]=d
   for y in years:duration_years.append(dict(site=s,variant=tag,larval_year=y,premoult_ln=d['premoult_ln_by_year'][y],premoult_na=d['premoult_na_by_year'][y],phenological_ln=d['phen_ln_by_year'].get(y)))
  durations[s]=variants
 table('thermal_durations_by_year.csv',duration_years)
 table('thermal_duration_summary.csv',[dict(site=s,variant=v,**{k:x for k,x in d.items() if not isinstance(x,dict)}) for s,ds in durations.items() for v,d in ds.items()])
 # Match every available mouse year to available weather; exclude 2026 explicitly.
 context=[];joint={}
 for s in SITES:
  yrs=sorted(y for (ss,y),ix in neon['index'].items() if ss==s and ix>0 and y+1 in cov[s+'_tick'])
  x=np.array([math.log(neon['index'][(s,y)]) for y in yrs]);w=np.array([project(cov[s+'_tick'][y+1],fit_neon) for y in yrs])
  dx=np.diff(np.array(yrs))==1
  joint[s]=dict(years=yrs,log_mouse=x.tolist(),weather=w.tolist(),sigma_mouse=float(x.std(ddof=1)),sd_weather=float(w.std(ddof=1)),rho=S.pearson_r(x,w),phi_mouse=S.pearson_r(x[:-1][dx],x[1:][dx]),phi_weather=S.pearson_r(w[:-1][dx],w[1:][dx]))
 for s in L.R14_ALL13:
  lab=s+'_tick' if s in SITES else s+'_tower';ys=sorted(y for ss,y in neon['index'] if ss==s and y in cov[lab])
  context.append(dict(site=s,sigma=neon['sigma'][s],r1=neon['r1'][s],matched_weather_years=len(ys),**{k:float(np.mean([cov[lab][y][k] for y in ys])) for k in L.COV_KEYS}))
 table('neon_13site_context.csv',context)
 rs=cary_sets['primary'];ys=sorted({r['mouse_year'] for r in rs});mx=np.array([np.mean([r['x'] for r in rs if r['mouse_year']==y]) for y in ys]);cw=np.array([cov['CARY'][y+1]['PC1_env'] for y in ys])
 joint['CARY']=dict(years=ys,log_mouse=mx.tolist(),weather=cw.tolist(),sigma_mouse=float(mx.std(ddof=1)),sd_weather=float(cw.std(ddof=1)),rho=S.pearson_r(mx,cw),phi_mouse=S.pearson_r(mx[:-1],mx[1:]),phi_weather=S.pearson_r(cw[:-1],cw[1:]))
 cary_sigma=float(np.mean([np.std([math.log(v) for v in vs.values() if v>0],ddof=1) for vs in cary['dry_mice'].values()]))
 # Save exact observed assay test counts for later simulation (no names or identifiers).
 dpath=RAW/'cary_lyme/dryad_d1c8046_forest_food_web_2018/Lyme_Study_Dataset.csv'
 with dpath.open(encoding='latin-1') as f:headers=next(csv.reader(f))
 print('CARY assay count columns',[h for h in headers if any(k in h.lower() for k in ['test','total','tfa','pcr','num'])],flush=True)
 res=dict(started=started,completed=now(),hashes_verified=len(verified),spatial=spatial,neon_sites=site_rows,neon_adjustments=partial,neon_correlations=correlations,neon_t3=t3,tower_sensitivity=dict(stat=tower_stat,pca=tower_fit),pca_neon=fit_neon,pca_cary=fit_cary,cary_slopes=cary_rows,cary_correlations=cary_corr,station_check=station_check,durations=durations,joint=joint,cary_sigma=cary_sigma,cary_pairs=cary_sets,neon_pairs=neon_pairs,context=context)
 save('field_analysis.json',res)
 with (HERE/'report.md').open('a') as f:
  f.write('\n## Field analysis checkpoint ('+now()+')\n\n')
  f.write('The registered field comparisons have completed. All supplied environmental manifest hashes matched. The original six-site rank correlation and within-site NEON slope, and both Cary sample counts and slopes, were reproduced before adjustment. The new outcomes below are post hoc relative to R12/R14 and pre-specified for envcov.\n\n')
  f.write(f"NEON: climate-adjusted partial rank correlation {partial[1]['r']:.4f}; negative-side p={partial[1]['p_negative']:.4f}; positive-side p={partial[1]['p_positive']:.4f}. Registered outcome: **{partial[1]['verdict']}**.\n\n")
  for per in ['primary','holdout']:
   a=next(r for r in cary_rows if r['period']==per and r['covariate']=='PC1_nymph')
   f.write(f"Cary {per}: adjusted slope {a['b']:.4f}, block-bootstrap SE {a['se']:.4f}, percentile interval [{a['ci_low']:.4f}, {a['ci_high']:.4f}]. Registered band compatible: {a['band_compatible']}. Outcome: **{a['verdict']}**.\n\n")
  f.write('Full numerical tables are in results/field_analysis.json and the companion CSV files. Thermal and masking analyses follow.\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['fields','theory','simulation','report']);a=ap.parse_args()
 if a.phase=='fields':fields()
 else:
  import continuation_analysis as C
  getattr(C,a.phase)()
