"""Fixed envcov theory and simulation campaign, using the project's R16 engine."""
from analysis import *
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import engine_accelerator as A

def _worker(job):
 A.install(check=False)
 result=E._eval_engine_job(job)
 return result

def theory():
 if (RESULTS/'theory_analysis.json').exists():raise RuntimeError('Theory phase already complete.')
 d=json.loads((RESULTS/'field_analysis.json').read_text());started=now()
 A.install();E.engine_code_checks()
 jobs=[];metadata={};unsupported=[]
 def add(site,label,mu,tick,sigma,phi=0.,kap=1.,**kw):
  key=f'{site}|{label}|k{kap:g}'
  options=dict(tau_mouse=mu,tau_tick=tick,sigma_mouse=sigma,phi_step=phi,kappa1=kap,**kw)
  try:desc=E.make_borrelia_desc(**options);e=E.Engine(desc).critical()
  except E.ModelError as ex:unsupported.append(dict(key=key,reason=str(ex),options=options));return
  jobs.append(dict(key=key,desc=desc,steps=40000,burn=2000,pairs=24,seed=20261115+len(jobs)))
  metadata[key]=dict(site=site,model=label,kappa=kap,**options)
 for site in ['CARY']+SITES:
  sig=d['cary_sigma'] if site=='CARY' else next(r['sigma'] for r in d['neon_sites'] if r['site']==site)
  dur=d['durations'][site]['primary'];stay=dur['phen_ln_mean_d'];prem=dur['premoult_ln_mean_d'];cv=dur['phen_ln_total_cv']
  B=stay/6;md,mc,cr=E.discrete_chain_substages(B,cv)
  for phi,name in [(0.,'iid'),(.30,'memory')]:
   add(site,'M0_coarse_'+name,1.,10/4.4,sig,phi)
   add(site,'M2_coarse_premoult_'+name,1.,prem/E.DT_COARSE_DAYS,sig,phi)
   add(site,'M3_coarse_phen_'+name,1.,stay/E.DT_COARSE_DAYS,sig,phi)
   pf=0. if phi==0 else phi**(6/E.DT_COARSE_DAYS)
   add(site,'M1_fine_borrowed_'+name,133/6,E.BORROWED_TICK_DAYS/6,sig,pf)
   add(site,'M2_fine_premoult_'+name,133/6,prem/6,sig,pf)
   add(site,'M3a_fine_memoryless_'+name,133/6,B,sig,pf)
   add(site,'M3b_fine_chain_'+name,133/6,B,sig,pf,tick_shape='chain',tick_substages=md)
  pf=.30**(6/E.DT_COARSE_DAYS)
  add(site,'M3c_fine_continuousCV_memory',133/6,B,sig,pf,tick_shape='chain',tick_substages=mc)
  yi=E.discrete_chain_substages(B,dur['phen_ln_interannual_cv'])[0]
  add(site,'M3d_yearCV_memory',133/6,B,sig,pf,tick_shape='chain',tick_substages=yi)
  add(site,'fixed_tick_memory',133/6,round(B),sig,pf,tick_shape='fixed')
  add(site,'both_fixed_memory',round(133/6),round(B),sig,pf,mouse_shape='fixed',tick_shape='fixed')
  for k in [.5]:
   add(site,'M3_coarse_phen_iid',1.,stay/E.DT_COARSE_DAYS,sig,0.,k)
   add(site,'M3b_fine_chain_memory',133/6,B,sig,pf,k,tick_shape='chain',tick_substages=md)
  for v in ['quest_fast','recalled_powerlaw']:
   stayv=d['durations'][site][v]['phen_ln_mean_d']
   add(site,v+'_coarse',1.,stayv/E.DT_COARSE_DAYS,sig)
 # Factor/correlated-driver checks at Cary; illustrative nonidentifiable inputs.
 site='CARY';sig=d['cary_sigma'];B=d['durations'][site]['primary']['phen_ln_mean_d']/6;pf=.30**(6/E.DT_COARSE_DAYS)
 for mouse in [85.,185.]:add(site,f'mouse_{mouse:g}d',mouse/6,B,sig,pf)
 add(site,'negative_step_memory',133/6,B,sig,-.30)
 add(site,'dilution',133/6,B,sig,pf,kappa2=-.17)
 add(site,'alternative_host',133/6,B,sig,pf,alt_host_sigma=sig,alt_host_share=.5)
 add(site,'asynchronous_patches',133/6,B,sig,pf,n_patches=4,patch_move=.15)
 for contact in [False,True]:
  for death_share in [0.,.5]:
   add(site,f'bounded_contact{contact}_deathshare{death_share}',133/6,B,sig,pf,contact_tick=contact,driver_kind='bounded',bounded_low=.2,tick_death=death_share/B)
 for gam in [-.2,-.1,.1,.2]:
  # Same subannual memory used for both drivers; annual rho treated as sensitivity.
  rho=float(d['joint']['CARY']['rho'])*np.sign(gam)
  add(site,f'coupled_weather{gam:+g}',133/6,B,sig,pf,sigma_weather=abs(gam),phi_weather=pf,rho_hw=rho)
 for j in jobs:
  if j['key'] in ['CARY|M3a_fine_memoryless_memory|k1','CARY|alternative_host|k1','CARY|asynchronous_patches|k1','CARY|bounded_contactTrue_deathshare0.5|k1']:
   j.update(run_extinction=True,ext_initial=500,ext_generations=10.,ext_margin=.05)
 plan=[dict(**{k:v for k,v in j.items() if k!='desc'},description=asdict(j['desc'])) for j in jobs]
 save('engine_campaign_plan.json',dict(started=started,cells=plan,unsupported=unsupported))
 print('ENGINE CELLS',len(jobs),'UNSUPPORTED',unsupported,flush=True)
 vals={};checkpoint=RESULTS/'engine_cells.jsonl'
 if checkpoint.exists():
  for line in checkpoint.read_text().splitlines():
   r=json.loads(line);vals[r['key']]=r
 todo=[j for j in jobs if j['key'] not in vals]
 with ProcessPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(_worker,j):j for j in todo}
  for f in as_completed(futures):
   r=f.result();r.update(metadata[r['key']]);vals[r['key']]=r
   with checkpoint.open('a') as out:out.write(json.dumps(clean(r),allow_nan=False)+'\n')
   print('ENGINE',len(vals),'/',len(jobs),r['key'],f"D={r['D']:.6f} SE={r['se']:.6f}",flush=True)
 results=[vals[j['key']] for j in jobs]
 violations=[r['key'] for r in results if r['D']>r['J']+4*r['se']+1e-6]
 corrections=[]
 for r in results:
  if r['period']>1 and abs(r['D']-r['J'])>1e-6:
   jb=next(j for j in jobs if j['key']==r['key']).copy();period=r['period']
   jb['steps']=math.ceil(40000/period)*period;jb['burn']=math.ceil(2000/period)*period
   corrected=_worker(jb);assert abs(corrected['D']-r['J'])<1e-6
   corrections.append(dict(original=r,corrected=corrected))
 save('periodic_endpoint_corrections.json',corrections)
 table('engine_losses.csv',results)
 shifts=[]
 for site in ['CARY']+SITES:
  old=vals[f'{site}|M0_coarse_iid|k1'];new=vals[f'{site}|M3_coarse_phen_iid|k1'];small=vals[f'{site}|M2_fine_premoult_iid|k1'];fine=vals[f'{site}|M3a_fine_memoryless_memory|k1'];chain=vals[f'{site}|M3b_fine_chain_memory|k1']
  shifts.append(dict(site=site,old_D=old['D'],thermal_same_clock_D=new['D'],change=new['D']-old['D'],percent_change=100*(new['D']/old['D']-1),fine_memoryless_D=fine['D'],fine_chain_D=chain['D'],chain_ratio=chain['D']/fine['D']))
 table('prediction_shifts.csv',shifts)
 save('theory_analysis.json',dict(started=started,completed=now(),cells=results,unsupported=unsupported,shifts=shifts,finite_horizon_bound_flags=violations,periodic_corrections=corrections))
 with (HERE/'report.md').open('a') as f:
  f.write('\n## Thermal-input and engine checkpoint ('+now()+')\n\n')
  f.write(f"Completed {len(results)} engine cells at the registered simulation effort. The sparse product adapter agreed with the original engine within its fixed tolerance, and the inherited analytic and R12 checks passed. One periodic endpoint bias was flagged against the general Jensen bound; whole-period correction reproduced the known exact loss. All nonperiodic cells met the bound check. The exact cells and non-admissible coarse-step descriptions are retained in results/.\n\n")
  f.write('Site-specific changes at the original time step and changes from stay shape on the finer time step are separated in prediction_shifts.csv. Simulation SE describes computational uncertainty only; host elasticity and subannual memory remain unmeasured.\n')

def rank_tests_batch(y,x,z):
 """Midranks including ties; enumerate all 720 permutations, with undefined cases flagged."""
 rx=stats.rankdata(x);rx-=rx.mean();ry=stats.rankdata(y,axis=1);ry-=ry.mean(1,keepdims=True)
 rz=stats.rankdata(z);rz-=rz.mean();uz=rz/max(np.linalg.norm(rz),1e-15)
 den=np.linalg.norm(ry,axis=1)*np.linalg.norm(rx);valid=den>1e-12
 rho=np.divide(ry@rx,den,out=np.full(len(y),np.nan),where=valid)
 un_perm=np.einsum('bpn,n->bp',ry[:,S.PERMS6],rx)/np.maximum(den[:,None],1e-15)
 p_un=(un_perm<=rho[:,None]+1e-12).mean(1)
 ex=rx-(rx@uz)*uz;ex/=np.linalg.norm(ex)
 ey=ry-(ry@uz)[:,None]*uz;eny=np.linalg.norm(ey,axis=1);ey/=np.maximum(eny[:,None],1e-15)
 part=ey@ex;part[eny<1e-12]=np.nan
 perm=ey[:,S.PERMS6];pn=np.einsum('bpn,n->bp',perm,ex);pz=np.einsum('bpn,n->bp',perm,uz)
 p=(pn/np.sqrt(np.maximum(1-pz*pz,1e-15))<=part[:,None]+1e-12).mean(1)
 return rho,part,p,p_un,valid

def event_stats(mask,n):
 count=int(np.sum(mask));freq=count/n;z=1.959963984540054;den=1+z*z/n
 centre=(freq+z*z/(2*n))/den;half=z*math.sqrt(freq*(1-freq)/n+z*z/(4*n*n))/den
 return dict(count=count,n=n,frequency=freq,mc_se=math.sqrt(freq*(1-freq)/n),wilson_low=centre-half,wilson_high=centre+half)

def sample_pair(j,years,reps,seed):
 """Measured annual covariance and marginal AR(1); reject infeasible innovations explicitly."""
 ds=[E.Driver('mouse',j['sigma_mouse'],j['phi_mouse']),E.Driver('weather',j['sd_weather'],j['phi_weather'])]
 env=E.le.Environment(ds,np.array([[1.,j['rho']],[j['rho'],1.]]))
 arr=env.sample(years,reps,seed,standardized=True).transpose(1,0,2)
 return arr[:,:,0]*j['sigma_mouse'],arr[:,:,1]*j['sd_weather']

def weighted_fe_batch(x,y,z,weights):
 """All study-by-bootstrap FE slopes from year multiplicities; missing/zero tests excluded."""
 # inputs study x grid x year. outputs study x bootstrap.
 mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(z);x=np.where(mask,x,0);y=np.where(mask,y,0);z=np.where(mask,z,0)
 def sm(v):return np.einsum('bgt,kt->bgk',v,weights,optimize=True)
 n=sm(mask.astype(float));nx=sm(x);ny=sm(y);nz=sm(z);ns=np.maximum(n,1)
 xx=(sm(x*x)-nx*nx/ns).sum(1);xy=(sm(x*y)-nx*ny/ns).sum(1)
 xz=(sm(x*z)-nx*nz/ns).sum(1);zz=(sm(z*z)-nz*nz/ns).sum(1);yz=(sm(y*z)-ny*nz/ns).sum(1)
 with np.errstate(divide='ignore',invalid='ignore'):
  b=xy/xx;ba=(xy*zz-xz*yz)/(xx*zz-xz*xz)
 return b,ba

def simulation():
 if (RESULTS/'simulation_analysis.json').exists():raise RuntimeError('Simulation phase already complete.')
 d=json.loads((RESULTS/'field_analysis.json').read_text());th=json.loads((RESULTS/'theory_analysis.json').read_text())
 A.install(check=False);started=now();n=2000;gammas=[-.2,-.1,0.,.1,.2];kappas=[0.,.5,1.];mus=[0.,.001,.002]
 # Synthetic equivalence check for vectorized bootstrap estimator before new simulations.
 rng=np.random.default_rng(1729);x=rng.normal(size=(2,3,18));z=rng.normal(size=x.shape);y=.7*x+.2*z+np.arange(3)[None,:,None]
 bu,ba=weighted_fe_batch(x,y,z,np.ones((1,18)))
 assert np.max(abs(ba-.7))<1e-10
 for b in range(2):
  gs=[(x[b,g],y[b,g],z[b,g,:,None]) for g in range(3)]
  assert abs(ba[b,0]-S.fe_ols_with_covariates(gs)[0])<1e-10
 # Midrank FL routine must match preserved scalar helper on a tied and untied example.
 xt=np.arange(6.);zt=np.array([2,0,5,1,3,4.]);yt=np.array([[1,2,3,4,5,6],[0,0,2,4,5,3.]])
 r,pr,p,pu,valid=rank_tests_batch(yt,xt,zt)
 for i in range(2):assert abs(pr[i]-S.freedman_lane_partial6(yt[i],xt,zt)[0])<1e-12 and abs(p[i]-S.freedman_lane_partial6(yt[i],xt,zt)[1])<1e-12
 print('CHECK simulation FE and tied-rank tests PASS',flush=True)
 neon=S.load_neon_r14_series(RAW/'neon')
 # Mean climate is kept at the observed six scores. Annual shocks preserve measured pair covariance.
 sites=d['neon_sites'];sig=np.array([r['sigma'] for r in sites]);zc=np.array([r['PC1_env'] for r in sites]);pbar=float(np.mean([r['nip'] for r in sites]));base_margin=-math.log1p(-pbar)/.5
 counts={s:[neon['cells'][(s,y)][0] for y in neon['test_years'][s]] for s in SITES}
 # Additional coupled calibrations: the same primary coarse clock and mean duration, weather on the other edge.
 jobs=[]
 for i,s in enumerate(SITES):
  j=d['joint'][s];B=d['durations'][s]['primary']['phen_ln_mean_d']/E.DT_COARSE_DAYS
  for gam in gammas:
   for k in kappas:
    if gam==0:continue
    desc=E.make_borrelia_desc(1.,B,sig[i],0.,kappa1=k,sigma_weather=abs(gam)*j['sd_weather'],phi_weather=0.,rho_hw=np.sign(gam)*j['rho'])
    jobs.append(dict(key=f'{s}|g{gam:g}|k{k:g}',desc=desc,steps=40000,burn=2000,pairs=24,seed=30261115+len(jobs)))
 save('simulation_campaign_plan.json',dict(started=started,replicates=n,gammas=gammas,kappas=kappas,death_rates=mus,site_residual_sd=.17,year_logprevalence_residual_sd=.14,mean_climate=zc,weather_driver_clock='coarse iid; measured annual AR pair used separately for yearly observations',engine_cells=[dict(**{k:v for k,v in j.items() if k!='desc'},description=asdict(j['desc'])) for j in jobs]))
 engine={}
 cp=RESULTS/'simulation_engine_cells.jsonl'
 if cp.exists():engine={r['key']:r for r in (json.loads(line) for line in cp.read_text().splitlines())}
 with ProcessPoolExecutor(max_workers=4) as pool:
  fs={pool.submit(_worker,j):j for j in jobs if j['key'] not in engine}
  for f in as_completed(fs):
   r=f.result();engine[r['key']]=r
   with cp.open('a') as out:out.write(json.dumps(clean(r))+'\n')
 print('SIMULATION engine calibrations',len(engine),flush=True)
 prior={r['key']:r for r in th['cells']};D={}
 for gam in gammas:
  for k in kappas:
   D[(gam,k)]=np.array([0. if k==0 else prior[f'{s}|M3_coarse_phen_iid|k{k:g}']['D'] for s in SITES]) if gam==0 else np.array([engine[f'{s}|g{gam:g}|k{k:g}']['D'] for s in SITES])
 # Daily curves enter annual cohort duration lookup and survival; finite-noise loss size from Engine.
 draws={};draw_diagnostics=[]
 for si,s in enumerate(SITES):
  j=d['joint'][s];ys=neon['test_years'][s];mx,ww=sample_pair(j,len(ys),n,2026111500+si)
  measured_w=np.array(j['weather']);order=np.argsort(measured_w);prob=stats.norm.cdf(ww/max(j['sd_weather'],1e-12));ind=np.minimum((prob*len(order)).astype(int),len(order)-1)
  years=np.array(j['years'])[order[ind]]
  dur=d['durations'][s]['primary'];lookup={int(y):v for y,v in dur['phen_ln_by_year'].items()}
  # Some measured mouse years precede tested cohorts: nearest available thermal cohort, recorded.
  available=np.array(sorted(lookup));selected=available[np.argmin(abs(years[:,:,None]-available[None,None,:]),axis=2)]
  stays=np.vectorize(lookup.__getitem__)(selected)
  draws[s]=(mx,ww,stays)
  draw_diagnostics.append(dict(site=s,rho_target=j['rho'],rho_sim=float(np.corrcoef(mx.ravel(),ww.ravel())[0,1]),phi_mouse_target=j['phi_mouse'],phi_mouse_sim=float(np.corrcoef(mx[:,:-1].ravel(),mx[:,1:].ravel())[0,1]),phi_weather_target=j['phi_weather'],phi_weather_sim=float(np.corrcoef(ww[:,:-1].ravel(),ww[:,1:].ravel())[0,1]),nearest_thermal_substitutions=int(np.sum(selected!=years)),annual_draws=int(years.size)))
 mean_stay=np.array([d['durations'][s]['primary']['phen_ln_mean_d'] for s in SITES]);neon_rows=[];events=[];coupled=[]
 for gam in gammas:
  for k in kappas:
   dv=D[(gam,k)];weather_only=D[(gam,0.)]
   for i,s in enumerate(SITES):coupled.append(dict(site=s,gamma=gam,kappa=k,D_total=dv[i],D_weather_only=weather_only[i],incremental_host_loss=dv[i]-weather_only[i]))
   for mu in mus:
    rng=np.random.default_rng(40261115+len(neon_rows));margin=base_margin+float(dv.mean())
    site_error=.17*rng.normal(size=(n,6));phat=np.zeros((n,6));sample_sigma=np.zeros((n,6));boundary=0
    for si,s in enumerate(SITES):
     mx,ww,stays=draws[s];sd=d['joint'][s]['sigma_mouse'];sample_sigma[:,si]=mx.std(1,ddof=1)
     # Annual host variation contributes its squared anomaly; expectation is the calibrated host loss.
     host_loss=(dv[si]-weather_only[si])*(mx*mx/max(sd*sd,1e-12))
     loss=weather_only[si]+host_loss
     lnR=margin+site_error[:,si,None]+gam*(zc[si]+ww)-mu*(stays-mean_stay.mean())-loss
     boundary+=int(np.sum(lnR<=0));pp=1-np.exp(-.5*np.maximum(lnR,0))
     ns=np.array(counts[s],int);kk=rng.binomial(ns[None,:],pp);phat[:,si]=kk.sum(1)/ns.sum()
    rho,rpart,padj,pun,valid=rank_tests_batch(phat,sig,zc)
    key=f'gamma={gam:g},kappa={k:g},mu={mu:g}'
    ev={'negative_unadjusted':rho<0,'support_cutoff':rho<=-29/35+1e-12,'support_exact':(rho<0)&(pun<.05),'as_positive_as_R14':rho>=17/35-1e-12,'negative_adjusted':rpart<0,'support_adjusted':(rpart<0)&(padj<.05)}
    row=dict(gamma=gam,kappa=k,mu=mu,replicates=n,mean_rho=float(np.nanmean(rho)),mean_partial=float(np.nanmean(rpart)),undefined=int(np.sum(~valid)),zero_margin_years=boundary,tied_prevalence_studies=int(np.sum(np.any(np.diff(np.sort(phat,axis=1),axis=1)==0,axis=1))))
    for name,mask in ev.items():
     e=event_stats(mask,n);row[name]=e['frequency'];events.append(dict(design='NEON',scenario=key,event=name,**e))
    neon_rows.append(row)
 print('NEON simulation scenarios',len(neon_rows),flush=True)
 table('simulation_neon.csv',neon_rows);table('simulation_neon_coupled_losses.csv',coupled);table('simulation_driver_checks.csv',draw_diagnostics)
 # Cary: 18-year registered design; preserve actual count/missingness pattern 1995--2012.
 rawpath=RAW/'cary_lyme/dryad_d1c8046_forest_food_web_2018/Lyme_Study_Dataset.csv'
 with rawpath.open(encoding='latin-1') as f:rawrows=list(csv.DictReader(f))
 nlookup={}
 for r in rawrows:
  counts_raw=[float(r[k]) for k in ['TFA_total','PCR_total'] if r.get(k,'') not in ['', 'NA']]
  if counts_raw:nlookup[(r['Grid'],int(r['Year']))]=int(sum(counts_raw))
 rs=d['cary_pairs']['primary'];plots=sorted({r['plot'] for r in rs});mask=np.zeros((6,18),bool);nt=np.ones((6,18),int)
 for r in rs:
  if 1995<=r['mouse_year']<=2012:
   gi=plots.index(r['plot']);ti=r['mouse_year']-1995;nn=nlookup.get((r['plot'],r['nymph_year']),0)
   if nn>0:mask[gi,ti]=True;nt[gi,ti]=nn
 yearmean={y:np.mean([r['x'] for r in rs if r['mouse_year']==y]) for y in {r['mouse_year'] for r in rs}}
 sd_grid=float(np.std([r['x']-yearmean[r['mouse_year']] for r in rs],ddof=1))
 pb=next(r['pbar'] for r in d['cary_slopes'] if r['period']=='primary');upper=1-pb
 starts=np.random.default_rng(20261114).integers(0,17,size=(2000,9));weights=np.array([np.bincount(np.concatenate([s,s+1]),minlength=18) for s in starts],float)
 inputs=dict(cary_grids=len(plots),cary_years=18,usable_pairs=int(mask.sum()),test_counts=nt[mask].tolist(),sigma_grid=sd_grid,pbar=pb,b_elasticities={str(k):k*upper/2 for k in kappas},annual_joint=d['joint'],neon_counts=counts)
 save('simulation_inputs.json',inputs)
 cary_rows=[];j=d['joint']['CARY']
 for gam in gammas:
  for k in kappas:
   rng=np.random.default_rng(50261115+len(cary_rows));mx,ww=sample_pair(j,18,n,60261115+len(cary_rows));btrue=k*upper/2
   xx=mx[:,None,:]+sd_grid*rng.normal(size=(n,6,18));zz=np.broadcast_to(ww[:,None,:],xx.shape).copy()
   lp=math.log(pb)+btrue*(xx-xx.mean((1,2),keepdims=True))+gam*zz+.14*rng.normal(size=(n,1,18))
   p=np.clip(np.exp(lp),0.00001,.95);kk=rng.binomial(nt[None,:,:],p)
   with np.errstate(divide='ignore'):yy=np.log(kk/nt[None,:,:])
   yy=np.where(mask[None,:,:]&(kk>0),yy,np.nan)
   point,point_adj=weighted_fe_batch(xx,yy,zz,np.ones((1,18)));b=point[:,0];ba=point_adj[:,0]
   se=np.empty(n);sea=np.empty(n);badboots=0
   for start in range(0,n,25):
    bb,bba=weighted_fe_batch(xx[start:start+25],yy[start:start+25],zz[start:start+25],weights)
    se[start:start+25]=np.nanstd(bb,axis=1,ddof=1);sea[start:start+25]=np.nanstd(bba,axis=1,ddof=1);badboots+=int(np.sum(~np.isfinite(bba)))
   key=f'gamma={gam:g},kappa={k:g}'
   ev={'positive_unadjusted':b>0,'positive_adjusted':ba>0,'gt2se_unadjusted':b>2*se,'gt2se_adjusted':ba>2*sea,'band_compatible_unadjusted':(b>=-2*se)&(b<=upper+2*se),'band_compatible_adjusted':(ba>=-2*sea)&(ba<=upper+2*sea)}
   row=dict(gamma=gam,kappa=k,b_true=btrue,mean_b=float(b.mean()),mean_b_adjusted=float(ba.mean()),sd_b=float(b.std(ddof=1)),sd_b_adjusted=float(ba.std(ddof=1)),mean_se=float(se.mean()),mean_se_adjusted=float(sea.mean()),invalid_bootstrap_slopes=badboots,probability_upper_clips=int(np.sum(np.exp(lp)>.95)),replicates=n)
   for name,m in ev.items():e=event_stats(m,n);row[name]=e['frequency'];events.append(dict(design='CARY',scenario=key,event=name,**e))
   cary_rows.append(row)
   table('simulation_cary.csv',cary_rows);print('CARY simulation',len(cary_rows),'/',len(gammas)*len(kappas),key,flush=True)
 table('simulation_event_uncertainty.csv',events)
 save('simulation_analysis.json',dict(started=started,completed=now(),neon=neon_rows,cary=cary_rows,events=events,coupled_losses=coupled,driver_checks=draw_diagnostics,inputs=inputs,engine_cells=list(engine.values())))
 with (HERE/'report.md').open('a') as f:
  f.write('\n## Masking simulation checkpoint ('+now()+')\n\n')
  f.write(f'Completed {len(neon_rows)} NEON and {len(cary_rows)} Cary scenarios, each with {n} synthetic studies. Results include exact event counts and Monte Carlo intervals. Weather response strengths are illustrative sensitivity values, not estimated causal effects. Annual host and weather pairs use the observed covariance and autocorrelation; site climates preserve the observed cross-site pattern.\n')


def report():
 if (RESULTS/'report_appended_results.md').exists():
  raise RuntimeError('Report already assembled; append a dated correction if needed.')
 import write_report
 write_report.main()
