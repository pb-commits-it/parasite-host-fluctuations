"""Summarize inherited simulator outputs without rerunning or replacing them."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
x=json.loads((a.RES/'theory_predictions.json').read_text())
r=x['Pa_fluctuating_transfers'];pa=[]
for z in r:
 if z['N_fluc']<3:
  n=next(t for t in r if t['sigma_hi']==z['sigma_hi'] and t['phi']==z['phi'] and t['N_fluc']==z['N_fluc']+1);pa.append(n['D']-z['D'])
pb=[];r=x['Pb_storage']
for z in r:
 if z['model'].startswith('storage'):
  no=next(t for t in r if t['sigma']==z['sigma'] and t['phi']==z['phi'] and t['model'].startswith('no_storage'));pb.append(z['D']-no['D'])
pc=[];r=x['Pc_spreading']
for z in r:
 if z['model'].startswith('generalist'):
  no=next(t for t in r if t['sigma']==z['sigma'] and t['model'].startswith('specialist'));pc.append(z['D']-no['D'])
pd=[];r=x['Pd_active_vs_trophic']
for z in r:
 if z['other_stay']==6 and z['phi']==.75 and z['d'] in [.05,.15]:
  no=next(t for t in r if t['other_stay']==6 and t['phi']==.75 and t['d']==1);pd.append(no['D']-z['D'])
out={'source':'inherited theory_predictions.json; numerical estimates read, not rerun','timestamp':a.stamp(),'P1_added_fluctuating_transfer':[min(pa),max(pa)],'P2_storage':[min(pb),max(pb)],'P3_alternative_hosts':[min(pc),max(pc)],'P4_active_vs_low_death_waiting_short_other_stay':[min(pd),max(pd)],'empirical_decline_scale':'unmeasured conversion from log reproduction loss to temporal count slope'}
a.save('theory_effect_sizes',out);print(json.dumps(out,indent=2))
