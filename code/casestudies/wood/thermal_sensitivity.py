#!/usr/bin/env python3
"""Registered illustrative Q10 timing sensitivity, NOT measured parasite rates.
Check formula against engine coefficient <=1e-10 before new calculations.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from theory_predictions import closed_form_B2,make_ring_description
sys.path.insert(0,str(a.HERE.parents[1]/'studies/engine-R16'))
import lifecycle_engine as le
import numpy as np

def main():
 checks=[]
 for tau in ([2.,3.,6.],[2.,3.,24.],[1.5,2.,12.]):
  for phi in [0.,.75]:
   c=2*closed_form_B2(tau,.6,phi)
   eng=le.Engine(make_ring_description(tau,[0,1],[.6,.6],[phi,phi])).critical().coefficient()['D2']
   err=abs(c-eng);assert err<1e-10;checks.append({'tau':tau,'phi':phi,'formula':c,'engine':eng,'error':err})
 mon=a.pd.read_csv(a.RES/'ersst_monthly.csv');mon=mon[(mon.grid_lat==48)&(mon.grid_lon==-124)]
 rows=[]
 for year,sub in mon.groupby('year'):
  temp=sub.sst_c.to_numpy();mt=temp.mean()
  for q in [1.5,2.,3.]:
   for clock in ['exposed_only','all_stages']:
    for mode in ['at_annual_mean_T','at_mean_monthly_rate']:
     factor=q**((mt-10)/10) if mode=='at_annual_mean_T' else np.mean(q**((temp-10)/10))
     for phi in [0.,.75]:
      vals=[]
      for b in [6.,24.]:
       tau=[2/factor,3/factor,b/factor if clock=='all_stages' else b]
       assert min(tau)>1
       d2=2*closed_form_B2(tau,.6,phi)
       vals.append(d2)
      rows.append({'year':int(year),'sst_mean_c':mt,'Q10':q,'temperature_exposure':clock,'rate_aggregation':mode,'host_phi':phi,'host_log_sd':.6,'development_rate_ratio':factor,'D2_no_storage':vals[0],'D2_storage':vals[1],'storage_ratio':vals[1]/vals[0],'storage_benefit':vals[0]-vals[1]})
 out=a.pd.DataFrame(rows);out.to_csv(a.RES/'thermal_sensitivity.csv',index=False)
 summary={'timestamp':a.stamp(),'engine_formula_checks':checks,'cases':len(out),'storage_order_reversals':int((out.storage_benefit<0).sum()),'development_rate_ratio_min':out.development_rate_ratio.min(),'development_rate_ratio_max':out.development_rate_ratio.max(),'storage_ratio_min':out.storage_ratio.min(),'storage_ratio_max':out.storage_ratio.max(),'deltaD_storage_min':out.storage_benefit.min(),'deltaD_storage_max':out.storage_benefit.max(),'no_storage_D2_range':[out.D2_no_storage.min(),out.D2_no_storage.max()],'storage_D2_range':[out.D2_storage.min(),out.D2_storage.max()],'caveat':'Illustrative Q10 values; fixed arithmetic-mean host transmission; no parasite thermal calibration, mortality response, feedback, or finite-noise inference.'}
 a.save('thermal_summary',summary);print(json.dumps(a.plain(summary),indent=2),flush=True)

if __name__=='__main__':main()
