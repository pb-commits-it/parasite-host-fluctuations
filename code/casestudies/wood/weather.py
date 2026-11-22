#!/usr/bin/env python3
"""Registered annual marine environment extension; prepare must precede analyse.
No inferred dates. ERSST cells are regional ocean proxies, not estuary readings.
"""
import sys,re,json,csv,math
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime,timedelta
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
import numpy as np
from scipy import stats
WRAW=a.RAW/'weather'

def haversine(lat,lon,la,lo):
 p1,p2=np.radians(lat),np.radians(la);dp=p2-p1;dl=np.radians(lo-lon)
 return 6371*2*np.arcsin(np.sqrt(np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2))

class Tables(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None
 def handle_starttag(self,tag,attrs):
  if tag=='tr':self.row=[]
  if tag in ('td','th'):self.cell=''
 def handle_data(self,s):
  if self.cell is not None:self.cell+=s
 def handle_endtag(self,tag):
  if tag in ('td','th') and self.cell is not None:
   if self.row is not None:self.row.append(self.cell.strip())
   self.cell=None
  if tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None

def prepare():
 # Explicit selection omits parasite outcomes.
 cols=['fish.id','year','site','lat','long','fish.spc','sfish_density','temp_na_rm','ord1','ord2']
 df=a.pd.read_csv(a.RAW/'wood_zenodo_7448240/final_dataset.csv',usecols=cols)
 fish=df.drop_duplicates('fish.id').copy()
 assert all(df.groupby('fish.id')[c].nunique().max()<=1 for c in cols if c!='fish.id')
 fish['date_start']=fish.year.astype(str)+'-01-01';fish['date_end']=fish.year.astype(str)+'-12-31'
 fish['date_precision']='year only';fish['coordinate_uncertainty']='unreported; some published coordinates geocoded from descriptive locality'
 fish['location_description']=np.where(fish.lat.isna(),'Puget Sound; exact collection locality unavailable in archive',fish.site)
 fish['source']='Wood Zenodo 7448240 final_dataset.csv'
 fish.to_csv(a.RES/'field_records.csv',index=False)
 raw=(WRAW/'ersst_v6_1880_2019.ascii').read_text()
 arr=np.full((1680,3,4),np.nan)
 for mt in re.finditer(r'^\[(\d+)\]\[(\d+)\], (.+)$',raw,re.M):
  arr[int(mt[1]),int(mt[2])]=[float(z) for z in mt[3].split(',')]
 arr[arr < -100]=np.nan
 times=[float(x) for x in raw.split('sst.time[1680]\n')[1].split('\n')[0].split(',')]
 dates=[datetime(1800,1,1)+timedelta(days=x) for x in times]
 assert dates[0].year==1880 and dates[-1].year==2019 and dates[-1].month==12
 lat=np.array([50.,48.,46.]);lon=np.array([-128.,-126.,-124.,-122.]);monthly=[];annual=[]
 for i,la in enumerate(lat):
  for j,lo in enumerate(lon):
   if not np.isfinite(arr[:,i,j]).any():continue
   for k,dt in enumerate(dates):monthly.append({'year':dt.year,'month':dt.month,'grid_lat':la,'grid_lon':lo,'sst_c':arr[k,i,j]})
   for yr in range(1880,2020):
    vals=arr[(yr-1880)*12:(yr-1879)*12,i,j]
    annual.append({'year':yr,'grid_lat':la,'grid_lon':lo,'sst_mean':np.mean(vals),'sst_monthly_sd':np.std(vals,ddof=1),'months':int(np.isfinite(vals).sum())})
 mon=a.pd.DataFrame(monthly);an=a.pd.DataFrame(annual)
 mon.to_csv(a.RES/'ersst_monthly.csv',index=False);an.to_csv(a.RES/'ersst_annual.csv',index=False)
 grid=an[['grid_lat','grid_lon']].drop_duplicates().to_numpy()
 fish['grid_lat']=np.nan;fish['grid_lon']=np.nan;fish['grid_distance_km']=np.nan
 for ix,r in fish.iterrows():
  if np.isfinite(r.lat) and np.isfinite(r['long']):
   dis=haversine(r.lat,r['long'],grid[:,0],grid[:,1]);ii=np.argmin(dis)
   fish.loc[ix,['grid_lat','grid_lon','grid_distance_km']]=[*grid[ii],dis[ii]]
 fish=fish.merge(an,on=['year','grid_lat','grid_lon'],how='left',validate='many_to_one')
 fish['race_rocks_distance_km']=haversine(fish.lat,fish['long'],48.2996,-123.5318)
 fish['sediment_PS1_distance_km']=haversine(fish.lat,fish['long'],47.347167,-122.409667)
 fish['sediment_PS4_distance_km']=haversine(fish.lat,fish['long'],47.614967,-122.449017)
 fish['weather_resolution']='ERSSTv6: 2 degree/month, aggregated annually; Race Rocks: point/annual'
 fish.to_csv(a.RES/'weather_matches.csv',index=False)
 parser=Tables();parser.feed((WRAW/'race_rocks_monthly.html').read_text());rr=[]
 for row in parser.rows:
  if len(row)>=13 and re.fullmatch(r'(19|20)\d\d',row[0]):
   yr=int(row[0])
   for month,x in enumerate(row[1:13],1):
    try:v=float(x)
    except ValueError:v=np.nan
    rr.append({'year':yr,'month':month,'sst_c':v})
 race=a.pd.DataFrame(rr).drop_duplicates(['year','month'])
 race.to_csv(a.RES/'race_rocks_monthly_refetched.csv',index=False)
 # Design-only diagnostics and illustrative pre-analysis power, one row/year.
 des=fish.dropna(subset=['sst_mean','sst_monthly_sd']).groupby('year').agg(sst_mean=('sst_mean','mean'),sst_monthly_sd=('sst_monthly_sd','mean'),n_fish=('fish.id','size')).reset_index()
 X=np.column_stack([np.ones(len(des)),stats.zscore(des.year),stats.zscore(des.sst_mean),stats.zscore(des.sst_monthly_sd)])
 w=des.n_fish.to_numpy()/des.n_fish.mean();V=.3**2*np.linalg.inv((X*w[:,None]).T@X)
 rng=np.random.default_rng(20261115);power=[]
 for j in [2,3]:
  for effect in [0.,.1,.3,.5]:
   beta=np.zeros(4);beta[j]=effect
   estimates=beta+rng.multivariate_normal(np.zeros(4),V,size=1000)
   rate=np.mean(abs(estimates[:,j]/np.sqrt(V[j,j]))>stats.t.ppf(.975,len(X)-4))
   power.append({'covariate':['sst_mean','sst_monthly_sd'][j-2],'effect':effect,'replicates':1000,'power_or_size':rate,'se':np.sqrt(V[j,j])})
 md={'timestamp':a.stamp(),'counts_not_read':True,'n_field_records':len(fish),'n_missing_coordinates':int(fish.lat.isna().sum()),'n_known_sites':int(fish[['lat','long']].dropna().drop_duplicates().shape[0]),'n_matched_fish':int(fish.sst_mean.notna().sum()),'n_matched_years':len(des),'year_min':int(fish.year.min()),'year_max':int(fish.year.max()),'grid_centers_used':fish[['grid_lat','grid_lon']].dropna().drop_duplicates().to_dict('records'),'grid_distance_km':fish.grid_distance_km.describe().to_dict(),'race_rocks_distance_km':fish.race_rocks_distance_km.describe().to_dict(),'design_rank':int(np.linalg.matrix_rank(X)),'design_condition':float(np.linalg.cond(X*np.sqrt(w[:,None]))),'year_sst_correlation':float(np.corrcoef(des.year,des.sst_mean)[0,1]),'year_monthly_sd_correlation':float(np.corrcoef(des.year,des.sst_monthly_sd)[0,1]),'illustrative_power':power,'race_refetch_year_min':int(race.year.min()),'race_refetch_year_max':int(race.year.max()),'missing_date_month_day':'all field records'}
 a.save('weather_pre_analysis',md);print(json.dumps(a.plain(md),indent=2),flush=True)

if __name__=='__main__':{'prepare':prepare}[sys.argv[1]]()
