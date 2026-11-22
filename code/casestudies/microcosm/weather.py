#!/usr/bin/env python3
"""Registered W1-W2 and labelled weather amendment diagnostics. No causal weather fit."""
from pathlib import Path
import sys,datetime,io,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *
LAKES={'Bishop':(42.5004,-83.8411),'Cedar':(42.3146,-84.079),'CrookedW':(42.3241,-84.1125),'LilAp':(42.5066,-83.8385),'Mill':(42.3287,-84.0905),'Walsh':(42.3375,-84.0798)}
FNAME={'LilAp':'LittleAppleton'}
def canonical(rows):
 for r in rows:
  if 'HostSpecies' in r:r['HostSpecies']=r['HostSpecies'].replace('_',' ')
  if 'Lake' in r:r['Lake']={'LittleAppleton':'LilAp','Little Appleton':'LilAp','Crooked W':'CrookedW'}.get(r['Lake'],r['Lake'])
 return rows
def dateparse(s):return datetime.datetime.strptime(s,'%m/%d/%y').date()
def hav(a,b):
 la,lo=np.radians(a);lb,lob=np.radians(b);return 6371*2*np.arcsin(np.sqrt(np.sin((lb-la)/2)**2+np.cos(la)*np.cos(lb)*np.sin((lob-lo)/2)**2))
def residual(x,Z):return x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]
def corr(x,y):
 x=np.asarray(x);y=np.asarray(y);ok=np.isfinite(x)&np.isfinite(y)
 if sum(ok)<3 or min(np.std(x[ok]),np.std(y[ok]))<1e-12:return np.nan
 return float(np.corrcoef(x[ok],y[ok])[0,1])
def main():
 SUMMARY.update(json.loads((OUT/'summary_metrics.json').read_text()));wx={};loc=[];joined=[]
 stations=[]
 for name,id in [('AnnArbor','USW00094889'),('Jackson','USW00014833')]:
  rr=read('weather',f'ghcnd_{name}_{id}_2021.csv');stations.append(dict(name=name,id=id,lat=num(rr[0]['LATITUDE']),lon=num(rr[0]['LONGITUDE']),records={datetime.date.fromisoformat(r['DATE']):r for r in rr}))
 for lake,(lat,lon) in LAKES.items():
  p=RAW/'weather'/f'daymet_{FNAME.get(lake,lake)}_2021.csv';txt=p.read_text();lines=txt.splitlines();header=next(i for i,s in enumerate(lines) if s.startswith('year,yday'))
  rr=list(csv.DictReader(lines[header:]));series={}
  for r in rr:
   day=datetime.date(int(r['year']),1,1)+datetime.timedelta(days=int(r['yday'])-1)
   series[day]={'tmean':(num(r['tmax (deg c)'])+num(r['tmin (deg c)']))/2,'prcp':num(r['prcp (mm/day)']),'srad':num(r['srad (W/m^2)']),'vp':num(r['vp (Pa)'])}
  wx[lake]={}
  for day in series:
   preceding=[day-datetime.timedelta(days=i) for i in range(1,15)]
   if all(d in series for d in preceding):wx[lake][day]={k:np.mean([series[d][k] for d in preceding]) for k in ['tmean','srad','prcp','vp']}
  nearest=min(stations,key=lambda s:hav((lat,lon),(s['lat'],s['lon'])))
  loc.append(dict(lake=lake,query_lat=lat,query_lon=lon,coordinate_status='inherited coordinates; exact sampling basin not verified; published lake/county location',source='Daymet V4R1',resolution_km=1,grid_centre_offset_km=None,grid_offset_status='not returned by single-pixel CSV; no invented exact distance',weather_start=min(series),weather_end=max(series),weather_days=len(series),nearest_station=nearest['id'],station_lat=nearest['lat'],station_lon=nearest['lon'],station_distance_from_query_km=hav((lat,lon),(nearest['lat'],nearest['lon']))))
  for day,w in wx[lake].items():
   nn=[nearest['records'].get(day-datetime.timedelta(days=i)) for i in range(1,15)];temp=[(num(r['TMAX'])+num(r['TMIN']))/2 for r in nn if r]
   joined.append(dict(lake=lake,date=day.isoformat(),day_of_year=day.timetuple().tm_yday,**w,ghcn_tmean=np.nanmean(temp),ghcn_n_days=sum(np.isfinite(temp))))
 save('weather_locations',loc);save('weather_daily_14d',joined)
 raw=canonical(read('davenport2024a_spores_outbreaks','Clean_Data_2021_Lakes_All_Host_Densities.csv'));records=[]
 for i,r in enumerate(raw,2):
  day=dateparse(r['Date']) if r['Date']!='NA' else datetime.date(int(r['Year']),1,1)+datetime.timedelta(days=int(r['Julian'])-1);w=wx[r['Lake']][day];records.append(dict(source_row=i,lake=r['Lake'],species=r['Host.Species'],date=day.isoformat(),day_of_year=day.timetuple().tm_yday,H=num(r['Host.Density']),total_examined=num(r['Total']),query_lat=LAKES[r['Lake']][0],query_lon=LAKES[r['Lake']][1],**w,**{k:num(r[k+'.prev']) for k in ['pasteuria','metsch','spiro','brood','micG']}))
 save('weather_field_records',records)
 # Record every distinct source row/date used from field data, including dated spore tables.
 all_dates=[]
 for p in sorted((RAW/'davenport2024a_spores_outbreaks').glob('*.csv')):
  rr=canonical(dh.read_csv_dicts(str(p)))
  for i,r in enumerate(rr,2):
   if r.get('Date')=='NA' and 'Julian' in r:r['Date']=(datetime.date(int(r['Year']),1,1)+datetime.timedelta(days=int(r['Julian'])-1)).strftime('%m/%d/%y')
   for key in ['Date','datetime','Date_outbreak_reached_1%','Date_Outbreak_Ends']:
    if key in r and r[key] and r[key]!='NA':
     day=dateparse(r[key]);lake=r['Lake'];all_dates.append(dict(file=p.name,row=i,date_field=key,date=day.isoformat(),lake=lake,lat=LAKES[lake][0],lon=LAKES[lake][1],date_matched=day in wx[lake],coordinate_status='approximate inherited query point'))
 save('field_record_dates',all_dates)
 locations=[]
 for (lake,),rr in groups(records,['lake']).items():locations.append(dict(lake=lake,host_rows=len(rr),dates=len(set(r['date'] for r in rr)),first=min(r['date'] for r in rr),last=max(r['date'] for r in rr),temperature_min=min(r['tmean'] for r in rr),temperature_max=max(r['tmean'] for r in rr)))
 save('field_date_coverage',locations)
 outbreak=canonical(read('davenport2024a_spores_outbreaks','SporeConcentrationBeginningVersusEnd.csv'));lag=canonical(read('davenport2024a_spores_outbreaks','AvgDaysBetweenSporeIncreaseAndOutbreakStart.csv'));lookup={(r['Lake'],r['HostSpecies'],r['Parasite']):r for r in outbreak};out=[]
 for r in lag:
  q=lookup[(r['Lake'],r['HostSpecies'],r['Parasite'])];day=dateparse(q['Date_outbreak_reached_1%']);w=wx[r['Lake']][day];out.append(dict(lake=r['Lake'],host=r['HostSpecies'],parasite=r['Parasite'],outbreak_date=day.isoformat(),signed_lag_days=num(r['DaysBetweenSporeIncreaseAndOutbreakStart']),elapsed_lead_days=-num(r['DaysBetweenSporeIncreaseAndOutbreakStart']),**w))
 save('w1a_outbreak_weather',out);a=np.array([r['signed_lag_days'] for r in out]);b=np.array([r['tmean'] for r in out]);sp=stats.spearmanr(a,b);pr=dh.perm_spearman(a,b,100000,SEED)
 record('W1a',dict(n=len(out),signed_rho=sp.statistic,signed_p_less=pr['p_less'],signed_p_greater=pr['p_greater'],elapsed_lead_rho=-sp.statistic,asymptotic_two_sided_p=sp.pvalue,lag_min=a.min(),lag_max=a.max()))
 # Primary pooled correlation and weather partial; lake/species + date are diagnostics.
 assoc=[];covars=[];host=[]
 for (lake,species),rr in groups(records,['lake','species']).items():
  rr=sorted(rr,key=lambda r:r['date']);host.append(dict(lake=lake,species=species,**host_stats([r['day_of_year'] for r in rr],[r['H'] for r in rr])))
 save('s6_host_memory',host)
 for parasite in ['pasteuria','metsch','spiro','brood','micG']:
  rr=[r for r in records if r['H']>0 and np.isfinite(r[parasite])];X=np.log([r['H'] for r in rr]);Y=np.array([r[parasite] for r in rr]);Z=np.array([[r['tmean'],r['srad']] for r in rr]);ids=[r['lake']+'|'+r['species'] for r in rr];cats=sorted(set(ids));dummy=np.array([[int(i==k) for k in cats[1:]] for i in ids]);doy=np.array([r['day_of_year'] for r in rr]);Zgroup=np.column_stack([np.ones(len(rr)),dummy]);rx=residual(X,Zgroup);ry=residual(Y,Zgroup)
  p=partial(X,Y,Z);plus=partial(X,Y,np.column_stack([Z,dummy,doy]));rawcorr=corr(X,Y)
  assoc.append(dict(parasite=parasite,n=len(rr),unadjusted_r=rawcorr,weather_partial_r=p['rho'],within_lake_species_r=corr(rx,ry),weather_group_date_partial_r=plus['rho'],changed_sign=rawcorr*p['rho']<0,halved=abs(p['rho'])<=.5*abs(rawcorr)))
  dev=rx**2
  for cov in ['tmean','srad','prcp','vp']:
   v=np.array([r[cov] for r in rr]);covars.append(dict(parasite=parasite,covariate=cov,n=len(rr),r_logH=corr(X,v),r_squared_logH_deviation=corr(dev,v),r_prevalence=corr(Y,v),within_group_r_prevalence=corr(ry,residual(v,Zgroup))))
 save('w1b_associations',assoc);save('weather_covariate_correlations',covars);record('W1b',assoc)
 # Daymet-vs-station sensitivity at the same field census dates.
 jj={(r['lake'],r['date']):r for r in joined};pairs=[(r['tmean'],jj[(r['lake'],r['date'])]['ghcn_tmean']) for r in records];pa=np.array(pairs)
 record('weather_coverage',dict(lakes=locations,field_rows=len(records),distinct_lake_dates=len(set((r['lake'],r['date']) for r in records)),dated_source_entries=len(all_dates),all_matched=all(r['date_matched'] for r in all_dates),daymet_ghcn_r=corr(pa[:,0],pa[:,1]),daymet_minus_ghcn_mean=np.nanmean(pa[:,0]-pa[:,1]),daymet_minus_ghcn_RMSE=np.sqrt(np.nanmean((pa[:,0]-pa[:,1])**2))))
 # Registered illustrative clock using C1, never a field development estimate.
 clock=next(c for c in SUMMARY['S5']['clock'] if c['origin']=='all' and c['metric']=='tti');theta=.8/(1-.8);ww=[]
 for r in records:
  for tout in [2,10,50]:
   for d in [1,.5,.1]:
    tin=np.exp(clock['intercept']+clock['slope']*r['tmean']);qout=1/(1+tout/theta);qin=1/(1+tin/theta);R=qout*qin;D=.5**2*(d*d*(1+2*theta)/(2*(tout+tin))-d*(R-(1-d)*qout)/(1-R))
    ww.append(dict(lake=r['lake'],date=r['date'],source_row=r['source_row'],tmean_C=r['tmean'],tau_in_proxy=tin,tau_out_illustrative=tout,d=d,sigma=.5,phi_day=.8,D2=D,extrapolated=not (10<=r['tmean']<=20)))
 save('w2_clock_sensitivity',ww);record('W2',dict(cells=len(ww),tau_proxy_min=min(r['tau_in_proxy'] for r in ww),tau_proxy_max=max(r['tau_in_proxy'] for r in ww),D2_min=min(r['D2'] for r in ww),D2_max=max(r['D2'] for r in ww),extrapolated_field_rows=sum(not (10<=r['tmean']<=20) for r in records),conclusion='Illustrative clock only; no validated air-to-water or cross-species development mapping'))
 record('weather_completed',now())
if __name__=='__main__':main()
