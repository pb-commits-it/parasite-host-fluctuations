"""Post hoc QC of refetched station copy; no model input is changed."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
import numpy as np
r=a.pd.read_csv(a.RES/'race_rocks_monthly_refetched.csv');bad=(r.sst_c < -2)|(r.sst_c>40);invalid=r[bad].to_dict('records');r.loc[bad,'sst_c']=np.nan
r.to_csv(a.RES/'race_rocks_monthly_qc.csv',index=False)
x=r.groupby('year').sst_c.agg(['mean','std','count']);x=x[x['count']>=11]
f=a.pd.read_csv(a.RES/'weather_matches.csv');c=f[['year','temp_na_rm']].dropna().drop_duplicates('year').set_index('year').join(x,how='inner');c['difference']=c.temp_na_rm-c['mean'];c.to_csv(a.RES/'race_archive_comparison_qc.csv')
s={'timestamp':a.stamp(),'post_hoc':True,'invalid_rows':invalid,'overlap_years':len(c),'mean_abs_difference':c.difference.abs().mean(),'max_abs_difference':c.difference.abs().max(),'monthly_sst_range_c':[r.sst_c.min(),r.sst_c.max()],'model_inputs_changed':False,'matched_year_min':int(f.loc[f.sst_mean.notna(),'year'].min()),'matched_year_max':int(f.loc[f.sst_mean.notna(),'year'].max()),'coordinates_lat_range':[f.lat.min(),f.lat.max()],'coordinates_lon_range':[f['long'].min(),f['long'].max()]}
a.save('weather_qc',s);print(json.dumps(a.plain(s),indent=2))
