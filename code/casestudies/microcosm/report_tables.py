#!/usr/bin/env python3
"""Print and archive report ranges, Monte Carlo precision and laboratory provenance."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *

def rows(name):return dh.read_csv_dicts(str(OUT/name))
def vals(rr,key):return [num(r[key]) for r in rr if np.isfinite(num(r[key]))]

def main():
 a=rows('s34_prediction_grid.csv');b=rows('s1_floquet.csv');h=rows('s6_host_memory.csv');c=rows('s5_clerc_spore_million.csv')
 extras=dict(timestamp=now(),S34_negative_periodic_cells=[r for r in a if num(r['D_Floquet'])<0],S1_D_by_d={d:dict(min=min(vals([r for r in b if num(r['d'])==d],'D')),max=max(vals([r for r in b if num(r['d'])==d],'D'))) for d in [1,.5,.1,.03]},S6=dict(series=len(h),finite_r1_series=len(vals(h,'r1')),r1_min=min(vals(h,'r1')),r1_max=max(vals(h,'r1')),zero_host_records=int(sum(vals(h,'zero_n')))),Clerc=dict(min_sacrifice_day=min(vals(c,'day')),max_sacrifice_day=max(vals(c,'day'))))
 (OUT/'report_extra_numbers.json').write_text(json.dumps(extras,indent=2)+'\n');print(json.dumps(extras,indent=2),flush=True)
 s=json.loads((OUT/'summary_metrics.json').read_text());mc=[]
 for name,r in s.items():
  if isinstance(r,dict) and r.get('permutations'):
   for tail in ['p_less','p_greater']:
    if tail in r:mc.append(dict(test=name,tail=tail,p=r[tail],draws=r['permutations'],MC_SE=math.sqrt(r[tail]*(1-r[tail])/r['permutations'])))
 save('monte_carlo_precision',mc)
 lab=[
 dict(study='S1 Ferris',temperature='28 C constant',light='Not found in examined methods',source='https://pmc.ncbi.nlm.nih.gov/articles/PMC7287369/',status='Temperature verified in methods; four-day resource cycle is not a light cycle'),
 dict(study='S2 Duncan',temperature='Daily 5/23 C white or red switching; extreme-temperature constant controls',light='Not verified',source='https://doi.org/10.1098/rspb.2013.1747',status='Treatment design verified; actual per-patch daily paths absent from supplied workbooks'),
 dict(study='S3 Dallas',temperature='Inherited 20 C claim not verified in examined primary methods',light='Not found; low microscope light during censuses is not an incubator photoperiod',source='https://taddallas.github.io/papers/Dallas2018RSOS.pdf',status='No quantitative thermal correction'),
 dict(study='S4 Dallas-Drake',temperature='Mean 20 C; 12/28 C pulses for 0/1/2/4 hours; filtering includes constant 12/20/28 C',light='12 h light / 12 h dark',source='https://pmc.ncbi.nlm.nih.gov/articles/PMC6093173/',status='Verified methods and deposited treatment labels'),
 dict(study='S5 Clerc',temperature='Inherited 20 C claim not independently verified',light='Not verified',source='https://pubmed.ncbi.nlm.nih.gov/25761710/',status='Abstract and data inspected; full-text challenge/timeout; no inferred light regime'),
 dict(study='S5 Tadiri',temperature='Constant 10/12.5/15/17.5/20 C',light='Culture methods: 16 h light / 8 h dark; treatment photoperiod not separately confirmed',source='https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/oik.10014',status='Temperature verified in data; culture regime in primary methods'),
 dict(study='S7 Krichel',temperature='20 C constant versus daily varying 14-26 C; 229 daily records',light='Not found in examined full XML methods',source='https://doi.org/10.1371/journal.pbio.3002260',status='Temperature verified in methods and daily file; host density absent')]
 save('laboratory_environment',lab)
 print(json.dumps(lab,indent=2),flush=True)

if __name__=='__main__':main()
