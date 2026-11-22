#!/usr/bin/env python3
"""Registered ML/REML taxon meta-regressions and 2,000-draw null bootstrap.
The M2 estimates are shrunken and their full cross-taxon covariance is not used
by the registered meta-regression. Unified count models are mandatory checks.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
import numpy as np
from scipy import optimize,stats,linalg
import json

def meta(y,X,v,groups=None,reml=False,start=None):
 n,p=X.shape
 if np.linalg.matrix_rank(X)<p: return {'estimable':False,'n':n,'p':p,'rank':int(np.linalg.matrix_rank(X))}
 if groups is None: Z=np.zeros((n,0))
 else: Z=np.eye(len(set(groups)))[np.unique(groups,return_inverse=True)[1]]
 def calc(q,extra=False):
  w=1/(v+q[0]); V=np.diag(w)
  if Z.shape[1]:
   WZ=w[:,None]*Z; h=1+q[1]*(Z.T@w)
   V-=((WZ*q[1]/h)@WZ.T)
   ld=np.log(v+q[0]).sum()+np.log(h).sum()
  else: ld=np.log(v+q[0]).sum()
  VX=V@X; A=X.T@VX; C=np.linalg.inv(A); beta=C@(VX.T@y)
  r=y-X@beta; vr=V@r
  f=.5*(ld+r@vr+(n-p if reml else n)*np.log(2*np.pi))
  P=V-VX@C@VX.T if reml else V
  if reml: f+=.5*np.linalg.slogdet(A)[1]
  g=[.5*(np.trace(P)-vr@vr)]
  if Z.shape[1]:g.append(.5*(np.sum(Z*(P@Z))-np.sum((Z.T@vr)**2)))
  if extra:return f,beta,C
  return f,np.array(g)
 q0=([.05,.01] if Z.shape[1] else [.05]) if start is None else start
 opt=optimize.minimize(calc,q0,jac=True,method='L-BFGS-B',bounds=[(0,4)]*len(q0),options={'ftol':1e-12,'gtol':1e-7,'maxiter':150})
 def pgmax(o):
  gr=calc(o.x)[1].copy()
  gr[(o.x<1e-8)&(gr>0)]=0
  gr[(o.x>4-1e-8)&(gr<0)]=0
  return float(max(abs(gr)))
 if pgmax(opt)>1e-5:
  alt=optimize.minimize(lambda q:calc(q)[0],opt.x,method='Powell',bounds=[(0,4)]*len(q0),options={'xtol':1e-11,'ftol':1e-12,'maxiter':150})
  refined=optimize.minimize(calc,alt.x,jac=True,method='L-BFGS-B',bounds=[(0,4)]*len(q0),options={'ftol':1e-14,'gtol':1e-8,'maxiter':200,'maxls':40})
  opt=min([opt,alt,refined],key=lambda o:calc(o.x)[0])
 f,b,C=calc(opt.x,True);se=np.sqrt(np.diag(C));df=n-p
 k=p+len(q0);aicc=2*f+2*k+2*k*(k+1)/(n-k-1)
 return {'estimable':True,'n':n,'p':p,'reml':reml,'nll':float(f),'aicc':float(aicc),'beta':b,'se':se,'ci_lo':b-stats.t.ppf(.975,df)*se,'ci_hi':b+stats.t.ppf(.975,df)*se,'pvalues':2*stats.t.sf(abs(b/se),df),'variance':opt.x,'converged':bool(pgmax(opt)<1e-4),'projected_gradient':pgmax(opt),'optimizer_success':bool(opt.success),'message':str(opt.message)}

def designs(tr):
 nh=tr.nhosts_wood_3plus.to_numpy();T=tr[a.TRAITS].to_numpy(float);N=np.column_stack([np.ones(len(tr)),nh==2,nh>=3]);return N,T

def holm(p):
 order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.maximum.accumulate(np.array(p)[order]*(len(p)-np.arange(len(p)))));return q

def main():
 gate=json.loads((a.RES/'reproduction_gate.json').read_text());assert gate['pass']
 df,tr=a.load();obs=a.pd.read_csv(a.RES/'M2_taxon_slopes.csv').set_index('para_spc').loc[tr.index]
 y=obs.slope.to_numpy();v=obs.se.to_numpy()**2;N,T=designs(tr);g=tr.par_type.to_numpy()
 # Known-result check: OLS and REML residual variance at homoscedastic v=0.
 rng=np.random.default_rng(23);xc=np.column_stack([np.ones(40),rng.normal(size=40)]);yc=xc@np.array([.3,-.4])+rng.normal(0,.2,40)
 chk=meta(yc,xc,np.zeros(40),None,True);bc=np.linalg.lstsq(xc,yc,rcond=None)[0];vc=np.sum((yc-xc@bc)**2)/38
 check={'ols_error':float(max(abs(chk['beta']-bc))),'reml_variance_error':float(abs(chk['variance'][0]-vc))}
 assert check['ols_error']<1e-10 and check['reml_variance_error']<1e-6
 a.save('meta_validation',check)
 nh=tr.nhosts_wood_3plus.to_numpy();idx=np.arange(len(tr))
 specs={'M3_nhosts_tax':(N,g,idx),'M5_traits_tax':(np.column_stack([np.ones(len(tr)),T]),g,idx),'M6_joint_notax':(np.column_stack([N,T]),None,idx),'M7_joint_tax':(np.column_stack([N,T]),g,idx),'M8_within_3plus':(np.column_stack([np.ones(sum(nh>=3)),T[nh>=3]]),g[nh>=3],idx[nh>=3])}
 # Registered trait coding and taxonomic sensitivities; no recoding.
 for name,col,j in [('S1_spreading_broad','spreading_broad',2),('S2_storage_alt','storage_alt_mollusc',1),('S3_waterborne','prop_waterborne_transfers',3),('S3_active_binary','has_active_freeliving',3)]:
  Ts=T.copy();Ts[:,j]=tr[col];specs[name]=(np.column_stack([N,Ts]),g,idx)
 valid=tr.terminal_in_fish.notna().to_numpy()
 specs['S2b_terminal']=(np.column_stack([N[valid],T[valid],tr.terminal_in_fish.to_numpy()[valid]]),g[valid],idx[valid])
 specs['S4_storage_active']=(np.column_stack([N,T,T[:,1]*T[:,3]]),g,idx)
 use=tr.resolved_taxon.to_numpy()==1;specs['S6_resolved']=(np.column_stack([N[use],T[use]]),g[use],idx[use])
 specs['S7_family']=(np.column_stack([N,T]),tr.taxonomic_family.to_numpy(),idx)
 out={}
 for name,(X,gg,ii) in specs.items():
  out[name]={s:meta(y[ii],X,v[ii],gg,s=='REML') for s in ['ML','REML']}
  if out[name]['REML']['estimable'] and X.shape[1]>=7:
   out[name]['REML']['holm_traits']=holm(out[name]['REML']['pvalues'][3:7])
  a.save('meta_models',out)
  print(name,a.plain(out[name]),flush=True)
 null=out['M3_nhosts_tax']['ML'];joint=out['M7_joint_tax']['ML'];lrt=2*(null['nll']-joint['nll']);delta=joint['aicc']-null['aicc']
 B=2000;rng=np.random.default_rng(20261114);gi=np.unique(g,return_inverse=True)[1];boot=[]
 for b in range(B):
  yy=N@null['beta']+np.sqrt(null['variance'][0]+v)*rng.normal(size=len(y))+np.sqrt(null['variance'][1])*rng.normal(size=max(gi)+1)[gi]
  nfit=meta(yy,N,v,g);jfit=meta(yy,np.column_stack([N,T]),v,g)
  boot.append([2*(nfit['nll']-jfit['nll']),nfit['converged'],jfit['converged']])
  if (b+1)%100==0:
   print('bootstrap',b+1,flush=True);a.save('meta_bootstrap_checkpoint',{'completed':b+1,'values':boot})
 pv=(1+sum(z[0]>=lrt for z in boot))/(B+1)
 rr=out['M7_joint_tax']['REML'];bs=rr['beta'][3:7];ps=rr['pvalues'][3:7];qh=holm(ps);sgn=np.array([-1,1,1,-1]);support=(bs*sgn>0)&(ps<.05);contr=(bs[:3]*sgn[:3]<0)&(qh[:3]<.05)
 decision='supports' if delta<=-2 and pv<.05 and any(support) else 'not decided'
 if delta<=-2 and any(contr):decision='contradicts'
 a.save('meta_decision',{'delta_aicc':delta,'lrt':lrt,'bootstrap_p':pv,'B':B,'bootstrap_failure_count':sum(not(z[1] and z[2]) for z in boot),'meta_only_classification':decision,'unified_confirmation_required':True,'traits_beta':bs,'traits_p':ps,'traits_holm':qh,'traits_ci_lo':rr['ci_lo'][3:7],'traits_ci_hi':rr['ci_hi'][3:7]})
 print('META DECISION',delta,lrt,pv,decision,flush=True)

if __name__=='__main__':main()
