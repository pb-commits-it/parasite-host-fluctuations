#!/usr/bin/env python3
"""Registered one-stage trait models and trait-coding sensitivities."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from meta_analysis import designs
import numpy as np

def warm(m,base,beta):
 vals=m.unpack_theta(m.initial_theta());out=[]
 for k,v in vals.items():
  if k=='beta_slope':v=beta
  elif k in base['pars']:v=base['pars'][k]
  out.extend(np.atleast_1d(v).tolist())
 return np.array(out)

def main():
 assert json.loads((a.RES/'reproduction_gate.json').read_text())['pass']
 df,tr=a.load();N,T=designs(tr);base=json.loads((a.RES/'M3_nhosts.json').read_text());b=np.array(base['pars']['beta_slope']);bn=np.array([b[0],b[1]-b[0],b[2]-b[0]])
 allidx=np.arange(len(tr));nh=tr.nhosts_wood_3plus.to_numpy()
 specs={'M3_nhosts_tax':(N,True,allidx,'par_type',bn),'M5_traits_tax':(np.column_stack([np.ones(len(tr)),T]),True,allidx,'par_type',np.r_[b.mean(),np.zeros(4)]),'M6_joint_notax':(np.column_stack([N,T]),False,allidx,'par_type',np.r_[bn,np.zeros(4)]),'M7_joint_tax':(np.column_stack([N,T]),True,allidx,'par_type',np.r_[bn,np.zeros(4)]),'M8_within_3plus':(np.column_stack([np.ones(sum(nh>=3)),T[nh>=3]]),True,allidx[nh>=3],'par_type',np.r_[b[2],np.zeros(4)])}
 for name,col,j in [('S1_spreading_broad','spreading_broad',2),('S2_storage_alt','storage_alt_mollusc',1),('S3_waterborne','prop_waterborne_transfers',3),('S3_active_binary','has_active_freeliving',3)]:
  Ts=T.copy();Ts[:,j]=tr[col];specs[name]=(np.column_stack([N,Ts]),True,allidx,'par_type',np.r_[bn,np.zeros(4)])
 valid=tr.terminal_in_fish.notna().to_numpy()
 specs['S2b_terminal']=(np.column_stack([N[valid],T[valid],tr.terminal_in_fish.to_numpy()[valid]]),True,allidx[valid],'par_type',np.r_[bn,np.zeros(5)])
 specs['S4_storage_active']=(np.column_stack([N,T,T[:,1]*T[:,3]]),True,allidx,'par_type',np.r_[bn,np.zeros(5)])
 use=tr.resolved_taxon.to_numpy()==1
 specs['S6_resolved']=(np.column_stack([N[use],T[use]]),True,allidx[use],'par_type',np.r_[bn,np.zeros(4)])
 specs['S7_family']=(np.column_stack([N,T]),True,allidx,'taxonomic_family',np.r_[bn,np.zeros(4)])
 summary={}
 for name,(X,tax,ii,tc,beta) in specs.items():
  if np.linalg.matrix_rank(X)<X.shape[1]:
   summary[name]={'estimable':False,'rank':int(np.linalg.matrix_rank(X)),'p':X.shape[1]};a.save('unified_trait_summary',summary);continue
  tri=tr.iloc[ii];sub=df[df['para.spc'].isin(tri.index)]
  d,nhsub=a.data_object(sub,tri,tc)
  # Keep original full-dataset units for all subset temporal effects.
  d.uyear=(sub.year.to_numpy()-df.year.mean())/df.year.std(ddof=1)
  d.slat=np.nan_to_num((sub.lat.to_numpy()-df.lat.mean())/df.lat.std(ddof=1))
  cats=np.unique(d.nh_of_obs,return_inverse=True)[1]
  m=a.LaplaceNBGLMM(d,X_slope=X,has_tax_re=tax,bog_obs_idx=cats,n_bog=len(set(cats)))
  r=a.fit_save('unified_'+name,m,warm(m,base,beta))
  offset=10;ps=X.shape[1]
  summary[name]={'estimable':True,'n_rows':len(sub),'n_taxa':len(tri),'aicc':r['aicc'],'nll':r['nll'],'beta':r['pars']['beta_slope'],'se':r['se_theta'][offset:offset+ps],'converged':r['converged'],'outer_grad_max':r['outer_grad_max'],'inner_grad_max':r['inner_grad_max'],'hessian_min_eigenvalue':r['hessian_min_eigenvalue']}
  a.save('unified_trait_summary',summary)
 print(json.dumps(a.plain(summary),indent=2),flush=True)

if __name__=='__main__': main()
