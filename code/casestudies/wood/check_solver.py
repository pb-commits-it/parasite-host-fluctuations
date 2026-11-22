#!/usr/bin/env python3
"""Check corrected full Hessian and Schur solve, plus saved reproduction optima.
Preset tolerance: directional derivative/Hessian relative error <1e-5,
Schur residual <1e-8, and saved optimum NLL agreement <1e-5.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
import numpy as np

def main():
 df,tr=a.load();d,nh=a.data_object(df,tr);m=a.LaplaceNBGLMM(d,X_slope=np.eye(3)[nh],bog_obs_idx=d.nh_of_obs,n_bog=3)
 old=json.loads((a.RES/'M3_nhosts.json').read_text());p=m.unpack_theta(np.array(old['theta']));w=np.array(old['w_hat']);nll,ww,_=m.inner_solve(p,w_init=w)
 f,g,H,V,z=m.joint_blocks(ww,p)
 rng=np.random.default_rng(411);v=rng.normal(size=len(w));v/=np.linalg.norm(v);eps=1e-4
 plus=m.joint_blocks(ww+eps*v,p);minus=m.joint_blocks(ww-eps*v,p)
 Hv=np.r_[H@v[:m.dim_v]+V@v[m.dim_v:],V.T@v[:m.dim_v]+z*v[m.dim_v:]]
 err=np.max(abs((plus[1]-minus[1])/(2*eps)-Hv))/max(1,np.max(abs(Hv)))
 step,_,_=m._schur_solve(v,H,V,z)
 res=np.r_[H@step[:m.dim_v]+V@step[m.dim_v:],V.T@step[:m.dim_v]+z*step[m.dim_v:]]+v
 nerr=abs(nll-old['nll']);sres=max(abs(res))
 out={'timestamp':a.stamp(),'nll_agreement':nerr,'hessian_directional_relative_error':err,'schur_solve_residual':sres,'inner_gradient':max(abs(g)),'pass':bool(nerr<1e-5 and err<1e-5 and sres<1e-8),'count_sum':int(df['count'].sum()),'nhost_taxon_counts':tr.nhosts_wood_3plus.value_counts().sort_index().to_dict(),'family_count':tr.taxonomic_family.nunique(),'resolved_taxa':int(tr.resolved_taxon.sum()),'missing_terminal_codes':int(tr.terminal_in_fish.isna().sum())}
 a.save('solver_check',out);print(json.dumps(a.plain(out),indent=2));assert out['pass']

if __name__=='__main__':main()
