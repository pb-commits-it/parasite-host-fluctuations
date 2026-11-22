"""Numerical audit: variance-component starts must agree in best objective.
No new substantive model or criterion. A difference >1e-5 requires repair.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from meta_analysis import meta,designs
import numpy as np
df,tr=a.load();obs=a.pd.read_csv(a.RES/'M2_taxon_slopes.csv').set_index('para_spc').loc[tr.index];y=obs.slope.to_numpy();v=obs.se.to_numpy()**2;N,T=designs(tr);g=tr.par_type.to_numpy();ix=np.arange(len(tr));nh=tr.nhosts_wood_3plus.to_numpy();use=tr.resolved_taxon.to_numpy()==1;terminal=tr.terminal_in_fish.notna().to_numpy()
spec={'M3_nhosts_tax':(N,g,ix),'M5_traits_tax':(np.column_stack([np.ones(len(tr)),T]),g,ix),'M7_joint_tax':(np.column_stack([N,T]),g,ix),'M8_within_3plus':(np.column_stack([np.ones(sum(nh>=3)),T[nh>=3]]),g[nh>=3],ix[nh>=3]),'S4_storage_active':(np.column_stack([N,T,T[:,1]*T[:,3]]),g,ix),'S6_resolved':(np.column_stack([N[use],T[use]]),g[use],ix[use]),'S2b_terminal':(np.column_stack([N[terminal],T[terminal],tr.terminal_in_fish.to_numpy()[terminal]]),g[terminal],ix[terminal])}
saved=json.loads((a.RES/'meta_models.json').read_text());results={}
for name,(X,gg,ii) in spec.items():
 for mode in ['ML','REML']:
  vals=[meta(y[ii],X,v[ii],gg,mode=='REML',start=q) for q in [[.01,0.],[.1,.1],[.01,1.],[.3,.3]]]
  best=min(vals,key=lambda z:z['nll']);old=saved[name][mode]['nll']
  results[name+'_'+mode]={'saved_nll':old,'best_multistart_nll':best['nll'],'improvement':old-best['nll'],'start_nlls':[z['nll'] for z in vals],'best':best}
  print(name,mode,old-best['nll'],flush=True)
a.save('meta_multistart_check',results)
