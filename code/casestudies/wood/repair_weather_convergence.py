"""Resume two iteration-limited fits at their saved modes; same likelihood/data.
Never accept an iteration-limit flag as convergence. Originals are retained.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from meta_analysis import designs
import numpy as np

def main():
 df,tr=a.load();cols=['temp_na_rm','ord1','ord2','sfish_density'];sub=df.dropna(subset=cols).copy();tri=tr.loc[sorted(sub['para.spc'].unique())];N,T=designs(tri);d,nh=a.data_object(sub,tri)
 d.uyear=(sub.year.to_numpy()-df.year.mean())/df.year.std(ddof=1);d.slat=np.nan_to_num((sub.lat.to_numpy()-df.lat.mean())/df.lat.std(ddof=1))
 E=np.column_stack([(sub[c]-sub[c].mean())/sub[c].std(ddof=1) for c in cols]);G=np.eye(3)[d.nh_of_obs];extra=np.column_stack([E[:,j]*G[:,k] for j in range(len(cols)) for k in range(3)])
 for label,X in [('baseline',N),('joint',np.column_stack([N,T]))]:
  name='S5_all_archived_'+label;path=a.RES/(name+'.json');old=json.loads(path.read_text())
  if old['converged']:continue
  attempt=1
  while (a.RES/(name+f'_attempt{attempt}_iteration_limit.json')).exists():attempt+=1
  path.rename(a.RES/(name+f'_attempt{attempt}_iteration_limit.json'))
  m=a.LaplaceNBGLMM(d,X_slope=X,has_tax_re=True,bog_obs_idx=d.nh_of_obs,n_bog=3,extra_fixed_X=extra);m._w_cache=np.array(old['w_hat'])
  start=np.array(old['theta'])
  # Boundary start is within the same declared optimizer bounds; the taxonomic
  # variance remains free. Its very flat descent caused the iteration limits.
  if old['pars']['logsigma_tax'] < -3:
   offset=0
   for key,val in m.unpack_theta(start).items():
    if key=='logsigma_tax':start[offset]=-7.;break
    offset+=np.size(val)
  r=a.fit_save(name,m,start)
  assert r['converged']

if __name__=='__main__':main()
