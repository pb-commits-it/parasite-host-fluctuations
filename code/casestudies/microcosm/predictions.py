#!/usr/bin/env python3
"""Bounded registered sensitivity campaign, conditional on clocks and host-law approximations.
Exact continuous telegraph dwell propagation; no dwell quantization or outcome fitting.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *
sys.path.insert(0,str(ROOT/'studies/engine-R16'))
import lifecycle_engine as le

def telegraph(tau_out,tau_in,k_in,d,scale,cycles=2000,pairs=12,seed=SEED):
 rng=np.random.default_rng(seed);hi=rng.exponential(scale,(cycles,pairs));lo=rng.exponential(scale,(cycles,pairs))
 # Paired path swaps states. Across each pair realized exposure mean is exactly one.
 durations=[np.concatenate([hi,lo],axis=1),np.concatenate([lo,hi],axis=1)];total=(durations[0][100:]+durations[1][100:]).sum(axis=0);n=1+k_in
 def rate(y):
  eig=[]
  for h in [1.8,.2]:
   Q=dh.build_ct_generator(tau_out,tau_in,1,k_in,d,h,y);w,V=linalg.eig(Q);W=linalg.inv(V);shift=max(w.real)
   # Verify spectral propagation locally against independent matrix exponential.
   probe=(V@np.diag(np.exp((w-shift)*scale))@W).real
   assert np.max(np.abs(probe-linalg.expm((Q-shift*np.eye(n))*scale)))<1e-7
   eig.append((w,V,W,shift))
  x=np.ones((2*pairs,n))/n;logs=np.zeros(2*pairs)
  for c in range(cycles):
   for state in range(2):
    dt=durations[state][c];w,V,W,shift=eig[state]
    x=((x@W.T)*np.exp(dt[:,None]*(w-shift)))@V.T;x=x.real
    if np.min(x)<-1e-8:raise ArithmeticError('spectral cancellation')
    x=np.maximum(x,0);s=x.sum(axis=1);x/=s[:,None]
    if c>=100:logs+=np.log(s)+shift*dt
  rates=logs/total;pm=(rates[:pairs]+rates[pairs:])/2
  return pm.mean(),pm
 root=optimize.brentq(lambda y:rate(y)[0],-.8,2,xtol=2e-6);r,pm=rate(root);dr=(rate(root+1e-4)[0]-r)/1e-4
 return root,pm.std(ddof=1)/np.sqrt(pairs)/abs(dr)

def markov_grid():
 hs=dh.read_csv_dicts(str(OUT/'s2_host_units.csv'));amps=[]
 # Use each uninfected series CV, then mean across the treatment; memory remains imposed driver memory.
 raw=xlsx('duncan2013_paramecium_holospora','Duncan_Paramecium_Time_7_13.xlsx',1)
 for key,rr in groups(raw,['Metapopulation']).items():
  if rr[0]['Infection']=='Uninfected' and rr[0]['Temperature']=='Variable':
   h=np.array([num(r['Data']) for r in rr]);amps.append(dict(id=key[0],noise=rr[0]['Red/White Variation'],CV=h.std(ddof=1)/h.mean()))
 amp=min(.9,np.mean([r['CV'] for r in amps]));save('s2_control_CV',amps);out=[]
 for ko,ki in [(1,1),(4,4),(1,16)]:
  for d in [1,.5,.1]:
   for arrest in [False,True]:
    for noise,phi in [('White',0),('Red',.8)]:
     D,se=dh.markov_daily_loss_D(2,7,ko,ki,d,phi,1+amp,1-amp,arrest_low=arrest,low_bg_loss=.02 if arrest else 0,horizon_days=2000,n_pairs=12)
     out.append(dict(noise=noise,phi_day=phi,k_out=ko,k_in=ki,d=d,thermal_arrest=arrest,amplitude=amp,D=D,se=se));print('S2 cell',out[-1],flush=True)
 save('s2_markov_predictions',out);record('S2_predictions',dict(cells=len(out),amplitude=amp,raw_mean_CV=np.mean([r['CV'] for r in amps]),min_D=min(r['D'] for r in out),max_D=max(r['D'] for r in out),actual_paths_available=False))
 return amp

def engine_grid(amp):
 # Daily clock cannot support stages 2/4 or 7/16; quarter-day/finer models would
 # change the model. Use dt=.25 throughout, with Erlang 4 only (minimum .5 d).
 dt=.25;out=[]
 for d in [1,.5,.1]:
  for k in [1,4]:
   for w in [0,.3,.6]:
    for phi in [0,.8]:
     for sync in [0,1]:
      for mig in [.0067,.0667]:
       shape='memoryless' if k==1 else 'chain'
       stages=[le.Stage('free'+str(j),mean=2/dt/k,death=d/(2/dt/k) if d<1 and j==k-1 else 0,contact=d<1 and j==k-1) for j in range(k)]+[le.Stage('host',mean=7/dt,shape=shape,substages=k)]
       # Same total host exit rate. A vertical bypass returns directly to host.
       transfers=[le.Transfer('free'+str(j),'free'+str(j+1)) for j in range(k-1)]+[le.Transfer('free'+str(k-1),'host',yield_=1/(1-d) if d<1 else 1,driver='H'),le.Transfer('host','free0',share=1-w,birth=True)]
       if w:transfers.append(le.Transfer('host','host',share=w,birth=True))
       try:
        desc=le.Description(stages=stages,transfers=transfers,drivers=[le.Driver('H',sigma=np.sqrt(np.log1p(amp**2)),phi=phi**dt,kind='bounded',low=None)],patches=le.Patches(count=2,synchrony=sync,mobile_stage='host',movement=1-(1-mig)**dt,topology='island'))
        e=le.Engine(desc);c=e.coefficient();out.append(dict(d=d,k=k,vertical_share=w,phi_day=phi,synchrony=sync,movement_day=mig,dt=dt,D2=c['D2'],levels=str(desc.drivers[0].levels()),status='computed'))
       except (le.ModelError,ValueError) as ex:out.append(dict(d=d,k=k,vertical_share=w,phi_day=phi,synchrony=sync,movement_day=mig,dt=dt,status=str(ex)))
 save('s2_engine_coefficients',out);record('S2_engine',dict(cells=len(out),computed=sum(r['status']=='computed' for r in out),interpretation='second-order coefficient, not exact finite-noise D; vertical self-loop is illustrative'))

def daphnia():
 m=SUMMARY['S4'];mu=m['nominal_mu'];F=m['nominal_F'];tau=m['nominal_tau_in'];h=160.;b=1/(mu+F*h/1000);d=mu*b;out=[]
 for k in [1,4,16]:
  for reading,dd in [('waste',1),('waiting',d)]:
   for scale in [7,14,28,56,112]:
    D=floquet(b,tau,k_in=k,d_wait=dd,period=scale);T,se=telegraph(b,tau,k,dd,scale)
    out.append(dict(k_in=k,reading=reading,d=dd,H_per_L=h,tau_out=b,tau_in=tau,scale_days=scale,amplitude=.8,D_Floquet=D,D_telegraph=T,se_telegraph=se));print('S3/S4 cell',out[-1],flush=True)
 save('s34_prediction_grid',out);record('S34_predictions',dict(cells=len(out),tau_out=b,tau_in=tau,d=d,min_Floquet=min(r['D_Floquet'] for r in out),max_Floquet=max(r['D_Floquet'] for r in out),min_telegraph=min(r['D_telegraph'] for r in out),max_telegraph=max(r['D_telegraph'] for r in out)))

def main():
 SUMMARY.update(json.loads((OUT/'summary_metrics.json').read_text()));record('predictions_started',now());amp=markov_grid();engine_grid(amp);daphnia();record('predictions_completed',now())
if __name__=='__main__':main()
