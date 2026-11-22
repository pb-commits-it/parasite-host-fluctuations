"""Registered Part 4 completion: weather changes Engine stage death AND residence time.

Build every environmental-state matrix with the project's Engine. Assemble their
exact arithmetic mean, retaining Gaussian host-weather dependence by truncated
normal moments. Use the project's loss root and matrix-product routine unchanged.
Annual block drivers repeat for 61 six-day steps; these are illustrative subannual
paths compatible with annual observations, not inferred fine-scale memory.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *
from continuation_analysis import event_stats,rank_tests_batch
from concurrent.futures import ProcessPoolExecutor,as_completed
import engine_accelerator as A
from scipy import linalg,special

class AnnualEnvironment:
 def __init__(self,j,sigma,kappa,order,host_prob):
  self.j=j;self.sigma=sigma;self.kappa=kappa;self.order=order;self.host_prob=host_prob;self.K=len(order)
  self.base=E.le.Environment([E.Driver('host',sigma,j['phi_mouse']),E.Driver('weather',1.,j['phi_weather'])],np.array([[1.,j['rho']],[j['rho'],1.]]))
 def sample(self,steps,replicates,seed,antithetic=False):
  phase=seed%61;ny=math.ceil((steps+phase)/61)
  z=self.base.sample(ny,replicates,seed,antithetic=antithetic,standardized=True)
  idx=np.minimum((special.ndtr(z[:,:,1])*self.K).astype(int),self.K-1)
  H=np.exp(self.kappa*self.sigma*z[:,:,0]-.5*(self.kappa*self.sigma)**2)
  nr=z.shape[1];rates=np.zeros((ny,nr,2*self.K))
  for k in range(self.K):
   rates[:,:,k]=(idx==k)*self.K
   rates[:,:,self.K+k]=(idx==k)*H/self.host_prob[k]
  return np.ascontiguousarray(np.repeat(rates,61,axis=0)[phase:phase+steps])

def make_model(job):
 j=job['joint'];sigma=job['sigma'];kap=job['kappa'];g=job['gamma'];dt=6.
 order=np.argsort(job['weather']);z=np.array(job['weather'])[order];dur=np.array(job['duration'])[order];K=len(z);q=1/K
 # Normal quantile bins map to actual paired year weather/duration states.
 cuts=special.ndtri(np.arange(K+1)/K);shift=kap*sigma*j['rho']
 qh=np.diff(special.ndtr(cuts-shift))
 comp=np.zeros((2,2*K,2,2));raw_matrices=[];state_r0=[]
 for k in range(K):
  death=1-math.exp(-.001*dt*math.exp(-g*(job['site_climate']+z[k])))
  desc=E.Description(stages=[E.Stage('mouse',mean=133/dt),E.Stage('tick',mean=dur[k]/dt,death=death)],transfers=[E.Transfer('mouse','tick',birth=True),E.Transfer('tick','mouse')])
  e=E.Engine(desc);parts=e.base_parts.copy();hostpart=np.zeros_like(parts);hostpart[0,0,1]=parts[0,0,1];parts[0,0,1]=0
  comp[:,k]=parts*q;comp[:,K+k]=hostpart*qh[k]
  raw_matrices.append(e.base_parts.copy());state_r0.append(e.R0)
 mean=comp.sum(1);U,F=mean
 R0=float(max(abs(linalg.eigvals(linalg.solve((np.eye(2)-U).T,F.T).T))))
 comp[1]/=R0;mean=comp.sum(1)
 base=E.Engine(E.make_borrelia_desc(133/6,float(np.mean(dur))/6,sigma,0.,kappa1=kap)).critical()
 base.base_parts=mean;base.rate_parts=comp;base.R0=1.;base.period=1;base.n=2
 lam,v,w=E.le._perron(mean.sum(0));assert abs(lam-1)<1e-10
 base.v=v;base.w=w;base.b=float(w@mean[1]@v)
 base.environment=AnnualEnvironment(j,sigma,kap,order,qh)
 base.rates=lambda x:x
 base.jensen=lambda **kw:float('nan') # one-hot decomposition has zero edges; no finite bound reported
 return base,dict(mean_R0=R0,state_R0=state_r0,weather_sorted=z.tolist(),duration_sorted=dur.tolist(),mean_tick_death_label='0.001/day with +/-0.2 climate sensitivity',exact_host_state_prob=qh.tolist(),raw_matrices=raw_matrices)

def worker(job):
 A.install(check=False);e,meta=make_model(job)
 # Check every state against independent matrix construction and a direct path product.
 rates=e.environment.sample(200,3,job['seed'],antithetic=True)
 logs,slopes=e._products(rates,.13,20,3)
 xx=np.tile(e.v,(6,1));direct=np.zeros(6)
 for t,rt in enumerate(rates):
  mat=e.base_parts[0]+math.exp(.13)*e.base_parts[1]+np.einsum('rk,kij->rij',rt-1,e.rate_parts[0]+math.exp(.13)*e.rate_parts[1])
  yy=np.einsum('rij,rj->ri',mat,xx);norm=yy.sum(1);xx=yy/norm[:,None]
  if t>=20:direct+=np.log(norm)
 direct/=180;direct=(direct[:3]+direct[3:])/2
 assert np.max(abs(logs-direct))<1e-10
 out=e.loss(steps=40000,burn=2000,pairs=24,seed=job['seed'])
 del meta['raw_matrices']
 return dict(site=job['site'],kappa=job['kappa'],gamma=job['gamma'],**out,**meta,product_check_maxerr=float(np.max(abs(logs-direct))))

def run():
 started=now();d=json.loads((RESULTS/'field_analysis.json').read_text());n=2000;jobs=[]
 for s in SITES:
  js=d['joint'][s];look=d['durations'][s]['primary']['phen_ln_by_year'];years=[y for y in js['years'] if str(y) in look]
  weather=[js['weather'][js['years'].index(y)]-np.mean(js['weather']) for y in years];duration=[look[str(y)] for y in years]
  sr=next(r for r in d['neon_sites'] if r['site']==s)
  for g in [-.2,0.,.2]:
   for k in [0.,.5,1.]:jobs.append(dict(site=s,joint=js,sigma=sr['sigma'],kappa=k,gamma=g,site_climate=sr['PC1_env'],weather=weather,duration=duration,seed=70261115+len(jobs)))
 save('driven_weather_plan.json',dict(started=started,jobs=jobs,steps=40000,burn=2000,pairs=24,replicates=2000,clock_days=6,annual_block_steps=61))
 cp=RESULTS/'driven_weather_cells.jsonl';results=[]
 if cp.exists():results=[json.loads(l) for l in cp.read_text().splitlines()]
 done={(r['site'],r['gamma'],r['kappa']) for r in results}
 with ProcessPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(worker,j) for j in jobs if (j['site'],j['gamma'],j['kappa']) not in done]
  for f in as_completed(fs):
   r=f.result();results.append(r)
   with cp.open('a') as out:out.write(json.dumps(clean(r))+'\n')
   print('DRIVEN',len(results),'/',len(jobs),r['site'],r['gamma'],r['kappa'],r['D'],r['se'],flush=True)
 sim=json.loads((RESULTS/'simulation_analysis.json').read_text());counts=sim['inputs']['neon_counts'];sites=d['neon_sites'];sigma=np.array([r['sigma'] for r in sites]);clim=np.array([r['PC1_env'] for r in sites]);pb=np.mean([r['nip'] for r in sites]);base=-math.log1p(-pb)/.5
 rows=[];events=[]
 for g in [-.2,0.,.2]:
  for k in [0.,.5,1.]:
   rng=np.random.default_rng(80261115+len(rows));cells=[next(r for r in results if r['site']==s and r['gamma']==g and r['kappa']==k) for s in SITES]
   D=np.array([r['D'] for r in cells]);lnR=np.log([r['mean_R0'] for r in cells]);margin=base-float(np.mean(lnR-D));res=.17*rng.normal(size=(n,6));phat=np.zeros((n,6))
   for i,s in enumerate(SITES):
    ns=np.array(counts[s]);state=np.log(cells[i]['state_R0']);ny=len(ns)
    # Observed-year environmental blocks preserve autocorrelated survival/development states.
    jj=rng.integers(0,len(state)-1,size=(n,(ny+1)//2));inds=np.stack([jj,jj+1],axis=2).reshape(n,-1)[:,:ny]
    # State arrays are ordered along climate, not time: use rank bins of the measured AR weather instead.
    pair=E.le.Environment([E.Driver('w',1,d['joint'][s]['phi_weather'])],np.ones((1,1)))
    wz=pair.sample(ny,n,90261115+len(rows)*6+i,standardized=True)[:,:,0].T
    inds=np.minimum((special.ndtr(wz)*len(state)).astype(int),len(state)-1)
    annual=margin+res[:,i,None]+state[inds]-D[i];p=1-np.exp(-.5*np.maximum(annual,0));kk=rng.binomial(ns[None,:],p);phat[:,i]=kk.sum(1)/ns.sum()
   rho,pr,padj,pun,valid=rank_tests_batch(phat,sigma,clim)
   row=dict(gamma=g,kappa=k,n=n,mean_rho=float(np.nanmean(rho)),mean_partial=float(np.nanmean(pr)))
   ev={'negative_unadjusted':rho<0,'support_cutoff':rho<=-29/35+1e-12,'support_exact':(rho<0)&(pun<.05),'as_positive_as_R14':rho>=17/35-1e-12,'negative_adjusted':pr<0,'support_adjusted':(pr<0)&(padj<.05)}
   for name,m in ev.items():e=event_stats(m,n);row[name]=e['frequency'];events.append(dict(gamma=g,kappa=k,event=name,**e))
   rows.append(row)
 table('driven_weather_sign_rates.csv',rows);table('driven_weather_event_uncertainty.csv',events)
 table('driven_weather_engine.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in results])
 save('driven_weather_analysis.json',dict(started=started,completed=now(),cells=results,scenarios=rows,events=events))
if __name__=='__main__':run()
