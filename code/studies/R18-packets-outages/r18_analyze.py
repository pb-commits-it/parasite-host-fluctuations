"""Descriptive analysis of the preregistered contrasts; no simulations."""
import json,math,pathlib
import numpy as np
from scipy.stats import spearmanr
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
rows=json.loads((P/'r18_cells.json').read_text())
cal=json.loads((P/'r18_calibration.json').read_text())
ix={(r['env'],r['a'],r['k'],r['b0'],r['L'],r.get('clump',False)):r for r in rows}

def find(env,a,k,b,L,clump=False):return ix.get((env,a,k,b,L,clump))
def diff(x,y,field='p'):
    sefield='se_gen' if field=='p_gen' else 'se'
    d=x[field]-y[field];se=math.hypot(x[sefield],y[sefield]);z=d/se if se else 0.
    return dict(first=x['key'],second=y['key'],d=d,se=se,z=z,resolved=abs(d)>4*se,visible=abs(d)>=.01)
contrasts={}
for field in ['p','p_gen']:
    cc=[]
    for env in ['A','B20','B200']:
      for a in [0,.5,1]:
       for k in [1,10,100]:
        for b in [1,100]:
         for L1,L2 in [(10,40),(40,160),(10,160)]:
          x=find(env,a,k,b,L1);y=find(env,a,k,b,L2)
          if x and y:cc.append(dict(env=env,a=a,k=k,b=b,stays=[L1,L2],**diff(x,y,field)))
    contrasts[field]=cc
for factor in ['share','packet','start','clump']:
    cc=[]
    for env in ['A','B20','B200']:
      for L in [10,40,160]:
       if factor=='share':pairs=[(find(env,1,k,b,L),find(env,0,k,b,L)) for k in [1,10,100] for b in [1,100]]
       elif factor=='packet':pairs=[(find(env,a,100,b,L),find(env,a,1,b,L)) for a in [0,.5,1] for b in [1,100]]
       elif factor=='start':pairs=[(find(env,a,k,100,L),find(env,a,k,1,L)) for a in [0,.5,1] for k in [1,10,100]]
       else:pairs=[(find(env,a,10,100,L,True),find(env,a,10,100,L)) for a in [0,1]]
       for x,y in pairs:
        if x and y:cc.append(dict(env=env,L=L,**diff(x,y)))
    contrasts[factor]=cc
for name,cc in contrasts.items():
    print(name,'comparisons',len(cc),'positive resolved',sum(x['resolved'] and x['d']>0 for x in cc),'negative resolved',sum(x['resolved'] and x['d']<0 for x in cc),'unresolved',sum(not x['resolved'] for x in cc))
for name in ['p','p_gen']:
    for pair in [[10,40],[40,160],[10,160]]:
        cc=[x for x in contrasts[name] if x['stays']==pair]
        print(name,pair,'benefit',sum(x['resolved'] and x['d']>0 for x in cc),'reversal',sum(x['resolved'] and x['d']<0 for x in cc),'unresolved',sum(not x['resolved'] for x in cc))
# D orderings across different laws, within one environment, with both differences resolved.
dmap={(r['env'],r['L']):r for rr in cal.values() for r in rr}
rev=[];con=[];tied=[]
for env in ['A','B20','B200']:
    rr=[r for r in rows if r['env']==env and r['kind']=='primary']
    for i,x in enumerate(rr):
      for y in rr[i+1:]:
        d=diff(x,y);cx=dmap[(env,x['L'])];cy=dmap[(env,y['L'])]
        dd=cx['D']-cy['D'];dse=math.hypot(cx['D_se'],cy['D_se'])
        if not d['resolved']:continue
        if x['L']==y['L']:tied.append(dict(env=env,**d))
        elif abs(dd)>4*dse:
            item=dict(env=env,D_first=cx['D'],D_second=cy['D'],**d)
            (con if dd*d['d']>0 else rev).append(item)
print('D resolved concordant',len(con),'reversed',len(rev),'same D different extinction',len(tied))
print('D strongest reversals',json.dumps(sorted(rev,key=lambda x:-abs(x['z']))[:8]))
# Descriptive survivor-minimum correlation; report the conditioning and confounding.
corr=[]
for a in [0,.5,1]:
    rr=[r for r in rows if r['kind']=='primary' and r['a']==a and r['survivor_host_min_quartiles']]
    if len(rr)>2:
        rho=float(spearmanr([r['p'] for r in rr],[r['survivor_host_min_quartiles'][1] for r in rr]).statistic)
        corr.append(dict(a=a,n=len(rr),rho=rho))
print('Survivor minimum correlations',corr)
print('Largest shared-death effects',json.dumps(sorted(contrasts['share'],key=lambda x:-abs(x['d']))[:6]))
print('Largest packet effects',json.dumps(sorted(contrasts['packet'],key=lambda x:-abs(x['d']))[:6]))
print('Largest start effects',json.dumps(sorted(contrasts['start'],key=lambda x:-abs(x['d']))[:6]))
print('Clump effects',json.dumps(contrasts['clump']))
(P/'r18_comparisons.json').write_text(json.dumps(dict(contrasts=contrasts,D_concordant=con,D_reversed=rev,D_tied=tied,correlations=corr),indent=2))
# All main estimates; each row is a full length comparison where present.
lines=[]
for env in ['A','B20','B200']:
    lines += [f'### {env}', '', '| Initial burden | Host-loss share | Packet | L10: p ± SE | L40: p ± SE | L160: p ± SE | Replicates, L10/40/160 |', '|---|---|---|---|---|---|---|']
    for b in [1,100]:
      for a in [0,.5,1]:
       for k in [1,10,100]:
        rr=[find(env,a,k,b,L) for L in [10,40,160]]
        if not any(rr):continue
        values=[f"{r['p']:.4f} ± {r['se']:.4f}" if r else 'not run' for r in rr]
        ns='/'.join(str(r['n']) if r else '—' for r in rr)
        lines.append(f'| {b} | {a} | {k} | '+' | '.join(values)+f' | {ns} |')
    lines += ['']
lines+=['### Ten nominal generations', '', 'The observation steps are 200, 500, and 1,700. Each entry is p ± SE.', '', '| Environment | Initial burden | Host-loss share | Packet | L10 | L40 | L160 |', '|---|---|---|---|---|---|---|']
for env in ['A','B20','B200']:
 for b in [1,100]:
  for a in [0,.5,1]:
   for k in [1,10,100]:
    rr=[find(env,a,k,b,L) for L in [10,40,160]]
    if not any(rr):continue
    values=[f"{r['p_gen']:.4f} ± {r['se_gen']:.4f}" if r else 'not run' for r in rr]
    lines.append(f'| {env} | {b} | {a} | {k} | '+' | '.join(values)+' |')
lines+=['','### Supplementary cells','','Each entry is calendar p ± SE. Fixed mean-10 comparators are in the primary tables.','','| Environment | Host-loss share | Packet law | L10 | L40 | L160 |','|---|---|---|---|---|---|']
for env in ['A','B20','B200','B5','B60']:
 for a in [0,1]:
  for k,clump in [(10,True),(1,False),(100,False)]:
   if not clump and env in ['A','B20','B200']:continue
   if clump and env in ['B5','B60']:continue
   rr=[find(env,a,k,100,L,clump) for L in [10,40,160]]
   if not any(rr):continue
   values=[f"{r['p']:.4f} ± {r['se']:.4f}" if r else 'not run' for r in rr]
   lines.append(f"| {env} | {a} | {'1/19, mean10' if clump else k} | "+' | '.join(values)+' |')
lines+=['','### Extinction times and occupied-host minima','','For every cell, the output and raw NPZ file contain extinction times and each path’s lowest occupied-host count. This table gives conditional medians. Time is conditional on extinction by 1,700. Host minimum is conditional on survival to 1,700. “—” means no observed extinction or no survivor, as applicable. The output also gives both quartiles.','','| Environment | Initial burden | Share | Packet | Extinction-time medians, L10/40/160 | Survivor host-minimum medians, L10/40/160 |','|---|---|---|---|---|---|']
for env in ['A','B20','B200']:
 for b in [1,100]:
  for a in [0,.5,1]:
   for k in [1,10,100]:
    rr=[find(env,a,k,b,L) for L in [10,40,160]]
    if not any(rr):continue
    def fmt(r,field):return f'{r[field][1]:g}' if r and r[field] else '—'
    ts='/'.join(fmt(r,'extinct_time_quartiles') for r in rr);ms='/'.join(fmt(r,'survivor_host_min_quartiles') for r in rr)
    lines.append(f'| {env} | {b} | {a} | {k} | {ts} | {ms} |')
(P/'r18_tables.md').write_text('\n'.join(lines)+'\n')
