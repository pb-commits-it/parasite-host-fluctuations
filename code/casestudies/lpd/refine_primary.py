"""Post-analysis numerical refinement registered in protocol dated amendment.
At most three direct coarse-cycle evaluations for each primary empirical regime.
No simulation samples, formula, bootstrap criterion, or empirical input are tuned.
"""
import sys,os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json,shutil,math
from study_io import *
from analysis import map_one
from numerics import Surfaces,direct
class Refined(Surfaces):
    def __init__(self):
        super().__init__('refinement_engine_aux_cache.json');self.target=None
        self.precision_path=RESULTS/'primary_direct_cells.json';self.cells=read(self.precision_path) if self.precision_path.exists() else {}
        old=read(RESULTS/'direct_engine_predictions.json')
        self.cache.update(old)
    def eval(self,key,sigma,phi,direct_outside=True):
        if not key.startswith('primary_coarse_') or phi!=self.target:return super().eval(key,sigma,phi,direct_outside)
        ck=f'{key}:{sigma:.12g}:{phi:.12g}'
        if ck not in self.cells:
            prior=super().eval(key,sigma,phi)
            if prior['method']=='direct_engine':d=prior
            else:d=direct('primary_coarse',self.data[key]['b'],sigma,phi)
            self.cells[ck]=dict(D=d['D'],se=d['se'],D_raw_linear=prior['D_raw_linear'],D_interpolated=prior['D'],method='direct_engine_refined',J=d['J'])
            if len(self.cells)%20==0:save(self.precision_path,self.cells)
        return self.cells[ck]
def run():
    initial=RESULTS/'primary_predictions.json.gz';archive=RESULTS/'primary_interpolation_predictions.json.gz'
    if not archive.exists():shutil.copyfile(initial,archive)
    f=RESULTS/'summary_tables.json';a=RESULTS/'summary_interpolation_tables.json'
    if f.exists() and not a.exists():shutil.copyfile(f,a)
    rows=read(archive);sf=Refined();dest=RESULTS/'primary_refinement_checkpoint.jsonl';done={}
    if dest.exists():
        for ln in dest.read_text().splitlines():
            r=json.loads(ln);done[r['series_id']]=r
    out=[]
    with dest.open('a') as file:
        for i,m in enumerate(rows):
            if m['series_id'] in done:r=done[m['series_id']]
            else:
                sf.target=m['r1_bc'];r=map_one(m,sf);r['extreme_lognormal_tail_flag']=m['sigma']>3.;file.write(json.dumps(clean(r))+'\n');file.flush()
            out.append(r)
            if i%100==0:print(now(),'direct primary',i,len(rows),'cells',len(sf.cells),flush=True)
    save(sf.precision_path,sf.cells);sf.flush();save(initial,out)
    save(RESULTS/'refinement_audit.json',dict(timestamp=now(),n_series=len(rows),n_cells=len(sf.cells),strict_long_changes=sum(a['long_iii_10pct']!=b['long_iii_10pct'] for a,b in zip(rows,out)),extreme_lognormal_series=sum(r['extreme_lognormal_tail_flag'] for r in out),changes_by_cycle={c:sum(a[c+'_iii_10pct']!=b[c+'_iii_10pct'] for a,b in zip(rows,out)) for c in ['long','moderate','short']}))
    print(now(),'refinement complete',flush=True)
if __name__=='__main__':run()
