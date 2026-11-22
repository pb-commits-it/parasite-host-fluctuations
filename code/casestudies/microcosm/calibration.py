#!/usr/bin/env python3
"""Registered S4-S7 extractions; assay fits are calibration, never fits to theory."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *
from scipy.special import gammaln, gamma

def hessian(fun,x):
 x=np.asarray(x);eps=1e-4;h=np.empty((len(x),len(x)))
 for i in range(len(x)):
  for j in range(len(x)):
   a=np.zeros(len(x));b=a.copy();a[i]=eps;b[j]=eps
   h[i,j]=(fun(x+a+b)-fun(x+a-b)-fun(x-a+b)+fun(x-a-b))/(4*eps*eps)
 return h

def s4():
 rr=read('dallas2016_metschnikowia_tempvar','sporeSurv.csv');fits=[];surv=[]
 for (trt,),rows in groups(rr,['trt']).items():
  t=np.array([num(r['day']) for r in rows]);dead=np.array([num(r['dead']) for r in rows]);n=np.array([num(r['total']) for r in rows]);ok=np.isfinite(t+dead+n)&(n>0);t=t[ok];dead=dead[ok];n=n[ok];live=n-dead
  for shape in ['exponential','Weibull']:
   def nll(v):
    lam=np.exp(v[0]);k=np.exp(v[1]) if len(v)>1 else 1.;logsurv=-(lam*t)**k
    return float(-np.sum(live*logsurv+dead*np.log(np.maximum(-np.expm1(logsurv),1e-300))))
   sol=optimize.minimize(nll,np.log([.05,1.] if shape=='Weibull' else [.05]),method='BFGS');pars=sol.x;cov=np.linalg.inv(hessian(nll,pars));se=np.sqrt(np.diag(cov));lam=np.exp(pars[0]);k=np.exp(pars[1]) if len(pars)>1 else 1.
   fits.append(dict(trt=trt,model=shape,FOV_n=len(t),spores_scored=int(n.sum()),lambda_per_day=lam,lambda_lo=np.exp(pars[0]-1.96*se[0]),lambda_hi=np.exp(pars[0]+1.96*se[0]),k=k,k_lo=np.exp(pars[1]-1.96*se[1]) if len(pars)>1 else 1.,k_hi=np.exp(pars[1]+1.96*se[1]) if len(pars)>1 else 1.,mean_viable_days=gamma(1+1/k)/lam,nll=nll(pars),AIC=2*len(pars)+2*nll(pars),optimizer_success=bool(sol.success)))
  for day in sorted(set(t)):
   nn=n[t==day].sum();ll=live[t==day].sum();surv.append(dict(trt=trt,day=day,live=ll,total=nn,fraction_viable=ll/nn))
 save('s4_spore_survival_fits',fits);save('s4_spore_survival_days',surv)
 filtering=[]
 for (trt,),rr in groups(read('dallas2016_metschnikowia_tempvar','filterRate.csv'),['trt']).items():filtering.append(dict(trt=trt,**mean_ci([num(r['mlDay']) for r in rr])))
 save('s4_filtering_ml_host_day',filtering)
 indiv=[];summ=[]
 for trt in [0,1,2,4]:
  rr=read('dallas2016_metschnikowia_tempvar',f'indivExpData{trt}.csv');infected=[r for r in rr if r['infected']=='1'];ages=[num(r['death.day']) for r in infected];yields=[np.mean([num(r[f'spores{i}']) for i in range(1,5)])*10*num(r['volume']) for r in infected]
  c=mean_ci(ages);cv=np.std(ages,ddof=1)/np.mean(ages)
  summ.append(dict(trt=trt,total_n=len(rr),infected_n=len(infected),mean_lifespan=c['mean'],se_lifespan=c['se'],lifespan_lo=c['lo'],lifespan_hi=c['hi'],cv=cv,k_moment=1/cv**2,spore_yield_mean=np.nanmean(yields),spore_yield_se=mean_ci(yields)['se']))
  for r,y in zip(infected,yields):indiv.append(dict(trt=trt,tube=r['tube'],death_day=num(r['death.day']),spore_yield=y))
 save('s4_infected_individuals',indiv);save('s4_host_duration',summ)
 mu=next(r['lambda_per_day'] for r in fits if r['trt']=='0' and r['model']=='exponential');F=next(r['mean'] for r in filtering if r['trt']=='20');tau=next(r['mean_lifespan'] for r in summ if r['trt']==0)
 contact=[]
 for h in [20,40,80,160,320,640]:contact.append(dict(H_per_L=h,mu_viability_day=mu,F_ml_host_day=F,encounter_per_day=F*h/1000,d=mu/(mu+F*h/1000),tau_out_contact_day=1/(mu+F*h/1000),tau_out_death_only_day=1/mu))
 save('s4_contact_curve',contact);record('S4',dict(survival=fits,filtering=filtering,host_duration=summ,nominal_mu=mu,nominal_F=F,nominal_tau_in=tau,contact_curve=contact))
 return mu,F,tau

def s5():
 r=read('clerc2015_pasteuria_stages','clerc_exp_1.csv');out=[]
 for key,rr in groups(r,['CLONE','SAMP_DPI']).items():out.append(dict(clone=key[0],day=num(key[1]),**mean_ci([num(q['spores_MIO_INDIV']) for q in rr])))
 save('s5_clerc_spore_million',out)
 rows=read('tadiri2023_pasteuria_epidemics','temp.csv');summ=[];jar=[]
 for r in rows:
  keys=sorted([k for k in r if k.startswith('t') and k[1:].isdigit()],key=lambda k:int(k[1:]));vals=[num(r[k]) for k in keys];times=[int(k[1:]) for k in keys]
  jar.append(dict(id=r['jar'],temp=num(r['Temp']),origin=r['de'],resistotype=r['Resistotype'],tti=num(r['tti']),htti=num(r['htti']),prev=num(r['prev']),**host_stats(times,vals)))
 for key,rr in groups(jar,['temp','origin','resistotype']).items():
  for metric in ['tti','htti','prev','mean_H','sd_logH','r1']:summ.append(dict(temp=key[0],origin=key[1],resistotype=key[2],metric=metric,**mean_ci([r[metric] for r in rr])))
 save('s5_tadiri_jars',jar);save('s5_tadiri_strata',summ)
 clocks=[]
 for origin in ['all','d','e']:
  for metric in ['tti','htti']:
   sub=[r for r in jar if (origin=='all' or r['origin']==origin) and np.isfinite(r[metric]) and r[metric]>0];x=np.array([r['temp'] for r in sub]);y=np.log([r[metric] for r in sub]);lr=stats.linregress(x,y);crit=stats.t.ppf(.975,len(x)-2)
   clocks.append(dict(origin=origin,metric=metric,n=len(sub),intercept=lr.intercept,slope=lr.slope,chi=-lr.slope,slope_se=lr.stderr,chi_lo=-lr.slope-crit*lr.stderr,chi_hi=-lr.slope+crit*lr.stderr,R_squared=lr.rvalue**2,min_temp=min(x),max_temp=max(x)))
 save('s5_temperature_clock',clocks)
 tempmeans=[]
 for (temp,),rr in groups(jar,['temp']).items():tempmeans.append(dict(temp=temp,**mean_ci([r['tti'] for r in rr if r['tti']>0])))
 save('s5_tti_temperature_means',tempmeans)
 record('S5',dict(clerc_records=len(r) if isinstance(r,list) else 297,jar_records=len(rows),clock=clocks,tempmeans=tempmeans,unidentifiable='Clerc individual maturation distribution; jar tti is visible infection, not development duration'))

def s7():
 rr=read('krichel2023_ordospora_tempvar','pbio.3002260.s016.csv');temp=read('krichel2023_ordospora_tempvar','pbio.3002260.s017.csv');pops=[];daily=[]
 daymap={int(num(r['obs_day'])):int(num(r['exp_day'])) for r in temp if np.isfinite(num(r['obs_day']))}
 for key,rows in groups(rr,['population','obs_day']).items():
  yes=[r for r in rows if r['inf_status'].strip().lower() in ['yes','no']];validburden=[num(r['n_clusters']) for r in rows]
  daily.append(dict(population=key[0],obs_day=int(key[1]),exp_day=daymap[int(key[1])],n=len(yes),prevalence=sum(r['inf_status'].strip().lower()=='yes' for r in yes)/len(yes) if yes else np.nan,mean_burden=np.nanmean(validburden)))
 for (pop,),rows in groups(daily,['population']).items():
  for window in ['full','endemic_day150']:
   sub=[r for r in rows if window=='full' or r['exp_day']>150];pops.append(dict(population=pop,variable=int(not pop.startswith('20C')),window=window,n_days=len(sub),n_observed_days=sum(np.isfinite(r['prevalence']) for r in sub),mean_prevalence=np.nanmean([r['prevalence'] for r in sub]),mean_burden=np.nanmean([r['mean_burden'] for r in sub])))
 save('s7_population_days',daily);save('s7_population_summary',pops)
 temps=[]
 for k in ['V1','V2','V3','V4','V5','V6']:
  v=np.array([num(r[k]) for r in temp]);temps.append(dict(population=k,n_days=len(v),mean_C=v.mean(),sd_C=v.std(ddof=1),min_C=v.min(),max_C=v.max(),r1=np.corrcoef(v[:-1],v[1:])[0,1]))
 save('s7_temperature_summary',temps);summary=[]
 for key,rows in groups(pops,['window','variable']).items():
  for m in ['mean_prevalence','mean_burden']:summary.append(dict(window=key[0],variable=key[1],metric=m,**mean_ci([r[m] for r in rows])))
 record('S7',dict(summary=summary,temperatures=temps,host_density='unrecorded'))

def main():
 SUMMARY.update(json.loads((OUT/'summary_metrics.json').read_text()));s4();s5();s7();record('completed_calibration',now())
if __name__=='__main__':main()
