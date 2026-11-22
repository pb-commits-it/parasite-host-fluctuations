#!/usr/bin/env python3
"""Registered microcosm analyses. Run python3 -I. No downloaded code is executed.
Benchmarks precede data analysis; original report and protocol are never rewritten.
"""
from pathlib import Path
import sys, os, csv, json, math, hashlib, subprocess, itertools
import numpy as np
from scipy import stats, optimize, linalg, integrate
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve(); ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
sys.path.insert(0,str(HERE))
import data_helpers as dh
RAW=ROOT/'data/raw/microcosm'; OUT=HERE/'results'; FIG=HERE/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
SEED=20261115
SUMMARY={}
def now():return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def floquet(tau_out,tau_in,k_out=1,k_in=1,d_wait=1,period=4,h_high=1.8,h_low=.2):
 def rate(y):
  n=k_out+k_in;M=np.eye(n);shift_sum=0.
  for h in [h_high,h_low]:
   Q=dh.build_ct_generator(tau_out,tau_in,k_out,k_in,d_wait,h,y);shift=max(linalg.eigvals(Q).real)
   M=linalg.expm((Q-shift*np.eye(n))*period/2)@M;shift_sum+=shift*period/2
  return (np.log(max(abs(linalg.eigvals(M))))+shift_sum)/period
 return float(optimize.brentq(rate,-2.5,6,xtol=1e-11))
def num(x):
 try:return float(x)
 except (TypeError,ValueError):return float('nan')
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.bool_):return bool(x)
 return x
def save(name,rows):
 rows=clean(rows)
 if rows:
  fields=list(dict.fromkeys(k for r in rows for k in r))
  with (OUT/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(name,'rows=',len(rows),flush=True)
def record(key,value):
 SUMMARY[key]=clean(value);(OUT/'summary_metrics.json').write_text(json.dumps(SUMMARY,indent=2,allow_nan=False)+'\n');print(key,json.dumps(clean(value)),flush=True)
def read(folder,file):return dh.read_csv_dicts(str(RAW/folder/file))
def xlsx(folder,file,sheet):
 rows=dh.read_xlsx_sheet(str(RAW/folder/file),sheet);return [dict(zip(rows[0],r+['']*max(0,len(rows[0])-len(r)))) for r in rows[1:]]
def groups(rows,keys):
 ans={}
 for r in rows:ans.setdefault(tuple(r[k] for k in keys),[]).append(r)
 return ans

def mean_ci(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)];n=len(v)
 if not n:return dict(n=0,mean=np.nan,se=np.nan,lo=np.nan,hi=np.nan)
 m=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan;q=stats.t.ppf(.975,n-1) if n>1 else np.nan
 return dict(n=n,mean=m,se=se,lo=m-q*se,hi=m+q*se)
def host_stats(t,h):
 t=np.asarray(t,float);h=np.asarray(h,float);ok=np.isfinite(h)&(h>0)
 ret={'n':len(h),'zero_n':int(np.sum(h==0)),'mean_H':float(np.nanmean(h))}
 z=np.full(len(h),np.nan);z[ok]=np.log(h[ok]);ret['sd_logH']=float(np.std(z[ok],ddof=1)) if ok.sum()>1 else np.nan
 if ok.sum()>2:
  resid=z[ok]-np.polyval(np.polyfit(t[ok],z[ok],1),t[ok]);ret['sd_logH_detrended']=float(resid.std(ddof=1))
 else:ret['sd_logH_detrended']=np.nan
 for lag in [1,2]:
  good=np.isfinite(z[:-lag])&np.isfinite(z[lag:]);a=z[:-lag][good];b=z[lag:][good]
  ret['r'+str(lag)]=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and min(np.std(a),np.std(b))>1e-12 else np.nan
 return ret

def label(res,expected=-1):
 v=res.get('diff',res.get('slope',res.get('rho',np.nan)))
 if v*expected>0 and res['p_less' if expected<0 else 'p_greater']<.05:return 'supports'
 if v*expected<0 and res['p_greater' if expected<0 else 'p_less']<.05:return 'contradicts'
 return 'not decided'
def slope(x,y,z=None):
 r=dh.perm_ols_slope(np.array(x),np.array(y),None if z is None else np.array(z),100000,SEED)
 df=len(x)-2-(0 if z is None else np.asarray(z).reshape(len(x),-1).shape[1]);r['se']*=np.sqrt((len(x)-(2 if z is None else 3))/df);q=stats.t.ppf(.975,df)
 r.update(lo=r['slope']-q*r['se'],hi=r['slope']+q*r['se'],n=len(x),permutations=100000,status=label(r));return r

def contrast(rows,key,group,one,blocks,expected=-1):
 missing=[r.get('id','') for r in rows if not np.isfinite(r[key])];rows=[r for r in rows if np.isfinite(r[key])]
 vals=np.array([r[key] for r in rows]);g=np.array([r[group]==one for r in rows]);ss=['|'.join(str(r[b]) for b in blocks) for r in rows]
 r=dh.stratified_diff_perm(vals,g,ss,10000,SEED)
 # Invert all balanced within-block label allocations exactly by meet-in-middle.
 sums=[]
 for block in sorted(set(ss)):
  ix=np.array([j for j,s in enumerate(ss) if s==block]);k=int(g[ix].sum())
  sums.append(np.array([vals[list(c)].sum() for c in itertools.combinations(ix,k)]))
 half=len(sums)//2
 def combine(parts):
  v=np.array([0.])
  for p in parts:v=(v[:,None]+p[None,:]).ravel()
  return v
 a=combine(sums[:half]);b=np.sort(combine(sums[half:]));target=vals[g].sum();total=len(a)*len(b)
 r['p_less_exact']=float(sum(np.searchsorted(b,target-u+1e-12,side='right') for u in a)/total)
 r['p_greater_exact']=float(sum(len(b)-np.searchsorted(b,target-u-1e-12,side='left') for u in a)/total);r['exact_allocations']=total;r['excluded_missing_ids']=missing
 # Stratified experimental-unit bootstrap, keeping each cell sample size.
 rng=np.random.default_rng(SEED);boot=np.zeros(10000)
 for block in sorted(set(ss)):
  for gp,sign in [(True,1),(False,-1)]:
   v=vals[(np.array(ss)==block)&(g==gp)];boot+=sign*rng.choice(v,(10000,len(v)),replace=True).sum(axis=1)/(g.sum() if gp else (~g).sum())
 r.update(lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),n=len(rows),permutations=10000,status=label(r,expected))
 return r

def partial(x,y,z,rank=False):
 x=np.asarray(x,float);y=np.asarray(y,float);z=np.asarray(z,float).reshape(len(x),-1);ok=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(z),axis=1);x=x[ok];y=y[ok];z=z[ok]
 if rank:x=stats.rankdata(x);y=stats.rankdata(y);z=np.column_stack([stats.rankdata(c) for c in z.T])
 X=np.column_stack([np.ones(len(x)),z]);a=x-X@np.linalg.lstsq(X,x,rcond=None)[0];b=y-X@np.linalg.lstsq(X,y,rcond=None)[0];r=float(np.corrcoef(a,b)[0,1]);df=len(x)-np.linalg.matrix_rank(X)-1
 p=float(stats.t.cdf(r*np.sqrt(df/max(1-r*r,1e-14)),df));delta=1.96/np.sqrt(max(len(x)-np.linalg.matrix_rank(X)-2,1));zz=np.arctanh(np.clip(r,-.999999,.999999))
 return dict(n=len(x),rho=r,p_less=p,p_greater=1-p,lo=float(np.tanh(zz-delta)),hi=float(np.tanh(zz+delta)))

def benchmarks():
 vals=[]
 for d,ref in [(1,.092043),(.5,.056857),(0,0.)]:
  v=dh.floquet_loss_D(10,5,d_wait=d,period=30);tol=1e-9 if d==0 else 5e-4;assert abs(v-ref)<tol;vals.append(dict(check='reference',d=d,value=v,tolerance=tol))
 # Independent scalar 2x2 eigenvalue at constant environment and ODE monodromy.
 for d in [1,.5,.1]:
  q=dh.build_ct_generator(10,5,1,1,d,1,0);assert abs(max(linalg.eigvals(q).real))<1e-9
  y=.12;M=np.eye(2)
  for h in [1.8,.2]:
   Q=dh.build_ct_generator(10,5,1,1,d,h,y)
   sol=integrate.solve_ivp(lambda t,v:(Q@v.reshape(2,2)).ravel(),(0,15),M.ravel(),rtol=1e-11,atol=1e-12);M=sol.y[:,-1].reshape(2,2)
  rate=np.log(max(abs(linalg.eigvals(M))))/30;ref=dh.floquet_growth(10,5,1,1,d,30,1.8,.2,y);assert abs(rate-ref)<1e-7
  vals.append(dict(check='independent_ODE',d=d,value=rate,reference=ref,error=abs(rate-ref),tolerance=1e-7))
 save('benchmarks',vals);record('benchmarks',{'passed':len(vals),'date':now()})

def s1():
 rows=xlsx('ferris2020_phage_oscillations','rspb20200787_si_002.xlsx',3)
 # Reverse numbering inferred from phase amplitudes and omitted Two = .9 RBG arm.
 amps={'One':.98,'Two':.9,'Three':.8,'Four':.5,'Five':0.}
 summary=[];unit=[];pred=[]
 for (trt,),rr in groups(rows,['Treatment (KB)']).items():
  kept=[r for r in rr if int(r['Transfer'])>0];lo=np.mean([num(r['Bacteria Density (CFU per ml)']) for r in kept if int(r['Transfer'])%6==3]);hi=np.mean([num(r['Bacteria Density (CFU per ml)']) for r in kept if int(r['Transfer'])%6==0]);hm=(hi+lo)/2
  summary.append(dict(treatment=trt,amplitude=amps[trt],low_phase_H=lo,high_phase_H=hi,mean_H=hm,delta_H=abs(hi-lo)/(2*hm)))
 base=next(r['mean_H'] for r in summary if r['amplitude']==0)
 for r in summary:
  r['mean_host_loss']=np.log(base/r['mean_H'])
  for d in [1,.5,.1,.03]:
   for ki in [1,16]:
    D=floquet(.35,.03,k_in=ki,d_wait=d,period=4,h_high=r['high_phase_H']/r['mean_H'],h_low=r['low_phase_H']/r['mean_H'])
    pred.append(dict(treatment=r['treatment'],amplitude=r['amplitude'],d=d,k_in=ki,D=D,mean_host_loss=r['mean_host_loss'],total_loss=D+r['mean_host_loss']))
    if d==1 and ki==1:r['D']=D;r['total_loss']=D+r['mean_host_loss']
 lookup={r['treatment']:r for r in summary}
 for (trt,rep),rr in groups(rows,['Treatment (KB)','Replicate']).items():
  rr=sorted(rr,key=lambda r:int(r['Transfer']));by={int(r['Transfer']):r for r in rr};kept=[r for r in rr if int(r['Transfer'])>0]
  detectable=[int(r['Transfer']) for r in rr if num(r['Phage denisty (PFU per ml)'])>0 or r['Phage amplified later?']=='Yes'];last=max(detectable)
  r=dict(treatment=trt,replicate=rep,amplitude=amps[trt],survival_transfer=last,right_censored=int(last==24),early_logPFU=np.mean([np.log1p(num(by[t]['Phage denisty (PFU per ml)'])) for t in [3,6]]),D=lookup[trt]['D'],total_loss=lookup[trt]['total_loss'],**host_stats([int(q['Transfer']) for q in kept],[num(q['Bacteria Density (CFU per ml)']) for q in kept]))
  unit.append(r)
 save('s1_units',unit);save('s1_host_treatments',summary);save('s1_floquet',pred)
 result=dh.perm_spearman(np.array([r['amplitude'] for r in unit]),np.array([r['survival_transfer'] for r in unit]),100000,SEED);result.update(status=label(result),n=len(unit),permutations=100000)
 record('T1a',result)
 y=[r['early_logPFU'] for r in unit]
 record('T1b_dynamic',slope([r['D'] for r in unit],y));record('T1b_total',slope([r['total_loss'] for r in unit],y));record('T1b_adjusted',slope([r['D'] for r in unit],y,np.log([r['mean_H'] for r in unit])))
 for r in summary:
  u=[v for v in unit if v['treatment']==r['treatment']];r.update({k+'_'+m:v for k,vs in [('early_logPFU',[q['early_logPFU'] for q in u]),('survival',[q['survival_transfer'] for q in u])] for m,v in mean_ci(vs).items()});r['persist_n']=sum(q['right_censored'] for q in u)
 save('s1_treatment_results',summary)
 # Per-unit endpoint resistance, avoiding thousands of clone assays as n.
 rbg=xlsx('ferris2020_phage_oscillations','rspb20200787_si_002.xlsx',4);out=[]
 for key,rr in groups(rbg,['Bacteria_Treatment','Bacteria_Replicate']).items():out.append(dict(treatment=key[0],replicate=key[1],mean_RBG=np.mean([np.clip(num(q['RBG']),0,1) for q in rr]),n_assays=len(rr)))
 save('s1_RBG_units',out)
 record('S1_summary',summary)

def s2():
 host=xlsx('duncan2013_paramecium_holospora','Duncan_Paramecium_Time_7_13.xlsx',1);par=xlsx('duncan2013_paramecium_holospora','Duncan_Parasite_time_07_13.xlsx',1)
 hs=[];hlookup={}
 for (ident,),rr in groups(host,['Metapopulation']).items():
  rr=sorted(rr,key=lambda r:int(r['Time']));r=rr[0];hlookup[ident]={int(v['Time']):num(v['Data']) for v in rr}
  hs.append(dict(id=ident,infection=r['Infection'],noise=r['Red/White Variation'],synchrony=r['Temperature synchrony'],migration=r['Migration'],temperature=r['Temperature'],**host_stats([int(v['Time']) for v in rr],[num(v['Data']) for v in rr])))
 save('s2_host_units',hs);hstat={r['id']:r for r in hs};units=[];long=[]
 for (ident,),rr in groups(par,['Metapopulation']).items():
  r=rr[0];by={int(v['Time']):v for v in rr};lp={t:np.log((num(v['Number infected'])+.5)/(num(v['Total'])+1)) for t,v in by.items()}
  u=dict(id=ident,noise=r['Red/White Variation'],synchrony=r['Temperature synchrony'],migration=r['Migration'],temperature=r['Temperature'],r016=(lp[16]-lp[0])/16,rI016=(lp[16]-lp[0]+np.log(hlookup[ident][16]/hlookup[ident][0]))/16,logp56=lp[56],late_logp=np.mean([p for t,p in lp.items() if t>=16]),mean_H=hstat[ident]['mean_H'])
  units.append(u)
  for t,v in by.items():long.append(dict(id=ident,noise=u['noise'],synchrony=u['synchrony'],temperature=u['temperature'],day=t,k=num(v['Number infected']),n=num(v['Total']),prev=num(v['Number infected'])/num(v['Total'])))
 save('s2_parasite_units',units);save('s2_prevalence',long);var=[r for r in units if r['temperature']=='Variable'];assert len(var)==32
 for tag,subset,key in [('T2a',var,'r016'),('T2a_infected_abundance',var,'rI016'),('T2a_synchronous',[r for r in var if r['synchrony']=='Synchronous'],'r016'),('T2b',var,'logp56')]:record(tag,contrast(subset,key,'noise','Red',['synchrony','migration']))
 red=[r for r in var if r['noise']=='Red']
 for key in ['r016','late_logp']:record('T2c_'+key,contrast(red,key,'synchrony','Asynchronous',['migration'],1))
 # Interaction: sign-flip outcome by noise then compare synchrony blocked by noise/migration.
 for key in ['r016','late_logp']:
  transformed=[dict(r,interact=r[key]*(1 if r['noise']=='Red' else -1)*2) for r in var]
  record('T2c_interaction_'+key,contrast(transformed,'interact','synchrony','Asynchronous',['noise','migration'],1))
 record('T2a_host_adjusted',slope([int(r['noise']=='Red') for r in var],[r['r016'] for r in var],np.column_stack([np.log([r['mean_H'] for r in var]),[int(r['synchrony']=='Synchronous') for r in var],[int(r['migration']=='High') for r in var]])))
 summaries=[]
 for key,rr in groups(hs,['infection','noise','synchrony','temperature']).items():
  for metric in ['mean_H','sd_logH','sd_logH_detrended','r1','r2']:summaries.append(dict(zip(['infection','noise','synchrony','temperature'],key),metric=metric,**mean_ci([r[metric] for r in rr])))
 save('s2_host_summary',summaries)
 traj=[]
 for key,rr in groups(long,['temperature','noise','day']).items():traj.append(dict(temperature=key[0],noise=key[1],day=key[2],**mean_ci([r['prev'] for r in rr])))
 save('s2_trajectories',traj)
 record('S2_sample',dict(host_units=len(hs),parasite_units=len(units),variable_units=len(var),max_day16_prevalence=max(r['prev'] for r in long if r['day']==16),mean_day16_prevalence=np.mean([r['prev'] for r in long if r['day']==16])))

def s3():
 raw=read('dallas2018_metschnikowia_threshold','expData.csv');units=[]
 for key,rr in groups(raw,['trt','popID']).items():
  rr=sorted(rr,key=lambda r:num(r['day']));days=[num(r['day']) for r in rr];h=[num(r['total'])*20 for r in rr];p=[num(r['prev']) for r in rr];d11=next(r for r in rr if num(r['day'])==11)
  units.append(dict(trt=int(key[0]),id=key[1],H0_L=int(key[0])*20,H11_L=num(d11['total'])*20,primary=int(num(d11['infected'])>0),primary_by11=int(any(num(r['infected'])>0 for r in rr if num(r['day'])<=11)),secondary=int(any(num(r['infected'])>0 for r in rr if num(r['day'])>18)),auc=float(np.trapezoid(p,days)),**host_stats(days,h)))
 save('s3_units',units);sup=[r for r in units if r['H0_L']>=160];assert len(sup)==45
 for rank in [False,True]:
  out=partial([r['sd_logH'] for r in sup],np.log([r['auc']+.01 for r in sup]),np.log([r['mean_H'] for r in sup]),rank);out['status']=('directionally consistent (observational)' if out['rho']<0 and out['p_less']<.05 else 'dominated by reverse-causal epidemic host suppression (registered label)' if out['rho']>0 and out['p_greater']<.05 else 'not decided');record('T3a_'+('Spearman' if rank else 'Pearson'),out)
 out=[]
 for (h,),rr in groups(units,['H0_L']).items():
  r=dict(H0_L=h,n=len(rr),mean_H11=np.mean([q['H11_L'] for q in rr]),mean_auc=np.mean([q['auc'] for q in rr]),mean_sd_logH=np.mean([q['sd_logH'] for q in rr]),mean_r1=np.nanmean([q['r1'] for q in rr]))
  for k in ['primary','secondary']:
   count=sum(q[k] for q in rr);ci=stats.binomtest(count,len(rr)).proportion_ci();r.update({k+'_n':count,k+'_fraction':count/len(rr),k+'_lo':ci.low,k+'_hi':ci.high})
  out.append(r)
 save('s3_threshold',out);record('S3_threshold',out)

def main():
 record('run',{'started':now(),'python':sys.version.split()[0],'numpy':np.__version__});benchmarks();s1();s2();s3();record('completed_primary',now())
 if '--primary-only' not in sys.argv:
  # Each module reloads the saved results; subprocess isolation prevents stale
  # imported SUMMARY state from overwriting a previous module's output.
  for module in ['calibration.py','predictions.py','weather.py','supplementary.py','report_tables.py','audit.py','figures.py']:
   print('Running',module,flush=True)
   subprocess.run([sys.executable,'-I',str(HERE/module)],check=True)
  print('Completed registered pipeline',now(),flush=True)
if __name__=='__main__':main()
