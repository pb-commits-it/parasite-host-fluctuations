"""Apply the already registered weather amendment with exact paired-date comparisons.
Outdoor grids are never assigned to laboratory records or used as ocean temperature.
"""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import datetime as dt,calendar,math,collections
import numpy as np
import helpers_continued as h
from weather_helpers import WeatherArchive,haversine_km
from analysis import candidates,eligible,metric,map_one,CYCLES
from study_io import *

def calendar_windows(s,mask):
    """Current and preceding physical-step weather, independent of host positivity.

    Preserve published predecessor periods whenever the full host grid contains
    them (including zero counts). Infer an absent predecessor from the registered
    sampling interval. A representative month is not necessarily a monthly step.
    """
    iv=s['interval_class'];spy=int(s['samples_per_year']);dates=s.get('step_dates_iso')
    def period(i):
        y=int(s['step_years'][i]);mo=int(s['step_months'][i])
        if not 1700<=y<=2025:return None
        if s['bin_days'] and dates:
            start=dt.date.fromisoformat(dates[i])
            return start,start+dt.timedelta(days=int(s['bin_days'])-1)
        if iv=='annual':return dt.date(y,1,1),dt.date(y,12,31)
        if s['database']=='GPDD' and 1<=int(s['subperiod'][i])<=12:
            return dt.date(y,mo,1),dt.date(y,mo,calendar.monthrange(y,mo)[1])
        if iv=='subannual_seasonal':
            width=12//spy;mm=((max(1,mo)-1)//width)*width+1;last=mm+width-1
            return dt.date(y,mm,1),dt.date(y,last,calendar.monthrange(y,last)[1])
        if not 1<=mo<=12:return None
        return dt.date(y,mo,1),dt.date(y,mo,calendar.monthrange(y,mo)[1])
    def previous(win):
        if win is None:return None
        start,end=win
        if s['bin_days'] and dates:
            delta=dt.timedelta(days=int(s['bin_days']))
            return start-delta,end-delta
        shift=12//spy
        def shifted(d,last=False):
            y,m=divmod(d.year*12+d.month-1-shift,12);m+=1
            return dt.date(y,m,calendar.monthrange(y,m)[1] if last else 1)
        if shifted(start).year<1700:return None
        return shifted(start),shifted(end,True)
    indices=np.flatnonzero(mask);all_positions={int(t):i for i,t in enumerate(s['steps'])}
    current=[period(int(i)) for i in indices];lag=[]
    for i,win in zip(indices,current):
        j=all_positions.get(int(s['steps'][i])-1)
        lag.append(period(j) if j is not None else previous(win))
    return current,lag

class MatchedArchive(WeatherArchive):
    def station_exact(self,lat,lon,windows,current_count=None):
        stations=sorted((haversine_km(lat,lon,la,lo),sid,la,lo) for sid,la,lo,_ in self.ghcn_stations if abs(la-lat)<=2.5)
        for dist,sid,la,lo in stations[:8]:
            if dist>250:continue
            sd=self.ghcn_data.get(sid,{})
            temps=np.full(len(windows),np.nan)
            for i,win in enumerate(windows):
                if win is None:continue
                start,end=win
                vals=[sd[(start.year,m)] for m in range(start.month,end.month+1) if (start.year,m) in sd]
                needed=8 if end.month-start.month==11 else math.ceil(.8*(end.month-start.month+1))
                if len(vals)>=needed:temps[i]=np.mean(vals)
            if np.isfinite(temps[:current_count]).sum()>=15:return dict(temps=temps,station_id=sid,slat=la,slon=lo,dist_km=dist)
        return None
    def match(self,s,mask):
        if s['environment_type']=='laboratory':return dict(matched=False,reason='laboratory; see reported regime'),[]
        lat,lon=s['latitude'],s['longitude']
        if not(-90<=lat<=90 and -180<=lon<=180):return dict(matched=False,reason='missing_coordinates'),[]
        yrs=s['step_years'][mask];mos=s['step_months'][mask];tt=s['steps'][mask];iv=s['interval_class'];realm=s['realm_detail']
        if ((yrs>=1700)&(yrs<=2025)).sum()<15:return dict(matched=False,reason='insufficient_calendar_dates'),[]
        # Daily coverage is preferred for subannual North American series.
        source=None;store=None;clat=clon=distance=None
        if iv!='annual':
            options=[]
            for k,v in self.daymet.items():
                dist=haversine_km(lat,lon,v['cell_lat'],v['cell_lon'])
                if dist<.1:options.append((len(v['by_date']),k,v))
            if options:
                _,key,store=max(options,key=lambda x:x[0]);source='Daymet_V4R1';clat=store['cell_lat'];clon=store['cell_lon'];distance=None
        if source is None and iv=='subannual_biweekly_weekly':return dict(matched=False,reason='monthly_weather_cannot_resolve_weekly_bin'),[]
        if source is None and realm=='Marine':
            cell=self.match_ersst_cell(lat,lon,int(np.median(yrs[yrs>=1700])))
            if cell is None:return dict(matched=False,reason='no_ocean_grid_within_250km'),[]
            ii,jj,clat,clon,distance=cell;source='NOAA_ERSST_v5';store=(ii,jj)
        if source is None:
            i=max(1,min(4318,int(round((90-lat)*24-.5))));j=max(1,min(8638,int(round((lon+180)*24-.5))))
            store=self.terraclimate.get((i,j))
            if store is not None:
                clat,clon=store['cell_lat'],store['cell_lon'];distance=haversine_km(lat,lon,clat,clon)
                if distance<=15:source='TerraClimate_V1.1'
        n=len(tt);T=np.full(2*n,np.nan);P=np.full(2*n,np.nan);records=[]
        current_windows,lag_windows=calendar_windows(s,mask)
        windows=current_windows+lag_windows
        for k in range(2*n):
            win=windows[k]
            if win is None:continue
            start,end=win;vts=[];vps=[];count=0;expect=0
            if source=='Daymet_V4R1':
                expect=(end-start).days+1
                for off in range(expect):
                    rec=store['by_date'].get(start+dt.timedelta(days=off))
                    if rec is not None:vts.append(rec[0]);vps.append(rec[1])
            elif source in ['TerraClimate_V1.1','NOAA_ERSST_v5']:
                expect=end.month-start.month+1
                for mo in range(start.month,end.month+1):
                    if source=='TerraClimate_V1.1':rec=store['ym'].get((start.year,mo))
                    else:
                        cube=self._get_ersst_year(start.year);ii,jj=store;val=int(cube[mo-1,ii,jj]) if cube is not None else -9999;rec=(val*.01,None) if val>-9990 else None
                    if rec is not None:
                        vts.append(rec[0])
                        if rec[1] is not None:vps.append(rec[1])
            if expect and len(vts)>=.8*expect:
                T[k]=np.mean(vts)
                if len(vps)>=.8*expect:P[k]=sum(vps)
                records.append(dict(series_id=s['series_id'],step=int(tt[k % n])-(k>=n),weather_role='lag' if k>=n else 'current',host_step=int(tt[k % n]),start_date=start.isoformat(),end_date=end.isoformat(),source=source,temperature_c=T[k],precipitation_mm=P[k],source_observations=len(vts),expected_observations=expect,site_lat=lat,site_lon=lon,matched_lat=clat,matched_lon=clon,distance_km=distance,distance_note='actual pixel centre unknown; request coordinates echoed' if source=='Daymet_V4R1' else 'grid-cell centre distance; site uncertainty additional'))
        if np.isfinite(T[:n]).sum()<15 and realm!='Marine' and iv!='subannual_biweekly_weekly':
            gh=self.station_exact(lat,lon,windows,current_count=n)
            if gh is not None:
                T=gh['temps'];P[:]=np.nan;source='NOAA_GHCNm_v4';clat,clon,distance=gh['slat'],gh['slon'],gh['dist_km'];records=[]
                for k in np.flatnonzero(np.isfinite(T)):
                    win=windows[k]
                    if win is None:continue
                    records.append(dict(series_id=s['series_id'],step=int(tt[k % n])-(k>=n),weather_role='lag' if k>=n else 'current',host_step=int(tt[k % n]),start_date=win[0].isoformat(),end_date=win[1].isoformat(),source=source,station_id=gh['station_id'],temperature_c=T[k],precipitation_mm=None,site_lat=lat,site_lon=lon,matched_lat=clat,matched_lon=clon,distance_km=distance,distance_note='station distance; site uncertainty additional'))
        ok=np.isfinite(T[:n])
        if ok.sum()<15:return dict(matched=False,reason='fewer_than_15_matched_steps',source=source,n_matched=int(ok.sum())),records
        # Do not compress missing years: lag is a physical grid step.
        cur=np.flatnonzero(ok & np.isfinite(T[n:]))
        if len(cur)<15:return dict(matched=False,reason='fewer_than_15_current_lag_pairs',source=source,n_matched=int(ok.sum())),records
        lag=cur+n;t=tt[cur].astype(float);y=np.log(s['values'][mask][cur]);tc=T[cur];pc=P[cur]
        if (np.diff(t)==1).sum()<12:return dict(matched=False,reason='fewer_than_12_adjacent_residual_pairs',source=source,n_matched=int(ok.sum())),records
        cols=[np.ones(len(t)),t-t.mean(),tc-tc.mean(),T[lag]-T[lag].mean()]
        useP=np.isfinite(pc).all() and np.std(pc)>1e-10
        if useP:cols.append(pc-pc.mean())
        X=np.column_stack(cols);Q=h.orthonormal_basis_from_design(X);res=y-Q@(Q.T@y)
        base=h.compute_core_metrics(t,y);m=h.compute_core_metrics(t,res);m['sigma2']=float(res@res/(len(t)-Q.shape[1]));m['sigma']=math.sqrt(m['sigma2'])
        m['mu0_r1']=h.projected_white_noise_r1_null_mean(t,Q);m['delta_r1']=m['r1']-m['mu0_r1'];m['r1_bc']=h.invert_projected_bias_corrected_r1(t,Q,m['r1'])
        base['mu0_r1']=h.white_noise_r1_null_mean(t);base['delta_r1']=base['r1']-base['mu0_r1'];base['r1_bc']=h.invert_bias_corrected_r1(t,base['r1'])
        mt=h.compute_core_metrics(t,tc);mt['r1_bc']=h.invert_bias_corrected_r1(t,mt['r1'])
        ey=h.ols_detrend(t,y)[0];et=h.ols_detrend(t,tc)[0]
        out=dict(matched=True,source=source,matched_lat=clat,matched_lon=clon,distance_km=distance,n_weather=int(ok.sum()),n_adjusted=len(t),matched_start_year=int(min(yrs[cur])),matched_end_year=int(max(yrs[cur])),mean_temp_c=float(tc.mean()),sigma_temp_c=mt['sigma'],r1_bc_temp=mt['r1_bc'],corr_host_temp=float(np.corrcoef(ey,et)[0,1]),r2_weather_explained=float(1-res@res/(ey@ey)),precipitation_in_design=bool(useP),residual_metrics=m,paired_host_metrics=base)
        if useP:
            mp=h.compute_core_metrics(t,pc);out['mean_prcp_mm']=float(pc.mean());out['sigma_prcp_mm']=mp['sigma'];out['r1_prcp']=mp['r1'];out['corr_host_prcp']=float(np.corrcoef(ey,h.ols_detrend(t,pc)[0])[0,1])
        # W3a: raw mean-stage duration, no clipping of inadmissible discrete stay.
        out['thermal_duration']={}
        for q10 in [2.,2.5,3.]:
            a=q10**((20-out['mean_temp_c'])/10);v={'a':a,'admissible_coarse':a>=1,'ratios':{c:b/a for c,b in CYCLES.items()}}
            for c,b in CYCLES.items():
                ph=base['r1_bc'];v[c+'_loss_penalty']=h.gamma0_two_stage(a,b,base['sigma'],ph)>base['sigma2']/(2*a) if a>=1 and ph>0 else False if a>=1 else None
            out['thermal_duration'][str(q10)]=v
        # W3b: exposed mean a=4 coarse observations, so thermal perturbations have positive staying.
        # The a=1 boundary formal coefficient is retained, explicitly not admissible as a clock model.
        ph=mt['r1_bc'];sig=mt['sigma']
        for c,b in CYCLES.items():
            for sign,pi in [('same',.05),('opposite',-.05)]:
                out['D2_thermal_'+sign+'_'+c]=h.thermal_loss_d2_one_exposed(4,b,sig,ph,pi,.08)
                out['D2_thermal_boundary_a1_'+sign+'_'+c]=h.thermal_loss_d2_one_exposed(1,b,sig,ph,pi,.08)
        return out,records

def run():
    primary=read(RESULTS/'primary_metrics.json');pm={r['series_id']:r for r in primary};ss=[s for s in candidates() if s['series_id'] in pm]
    land=list({(s['latitude'],s['longitude']) for s in ss if s['realm_detail']!='Marine' and math.isfinite(s['latitude']) and math.isfinite(s['longitude'])})
    print(now(),'loading weather',len(land),flush=True);a=MatchedArchive(RAW/'weather',land)
    print(now(),'loaded',len(a.terraclimate),len(a.daymet),len(a.ghcn_stations),flush=True)
    output=[];paired=[];adjusted=[];recall=[]
    for i,s in enumerate(ss):
        mask=s['values']>0;w,records=a.match(s,mask);recall.extend(records);w.update(series_id=s['series_id'],study_id=s['study_id'],realm=s['realm_detail'],interval_class=s['interval_class'],taxon_group=s['taxon_group'],latitude_band=pm[s['series_id']]['latitude_band']);output.append(w)
        if w['matched']:
            for key,target in [('residual_metrics',adjusted),('paired_host_metrics',paired)]:
                r=pm[s['series_id']].copy();r.update(w[key]);r['weather_source']=w['source'];r['mode']=key
                for lag,v in r['acf'].items():r['acf_'+str(lag)]=v
                target.append(r)
        if i%200==0:print(now(),'weather',i,'matched',len(adjusted),flush=True)
    save(RESULTS/'weather_metrics.json',output);save(RESULTS/'weather_adjusted_metrics.json',adjusted);save(RESULTS/'weather_paired_metrics.json',paired);csvwrite(RESULTS/'weather_record_matches.csv',recall)
    # Mechanistic coefficients are model predictions; no parasite observation is manufactured.
    print(now(),'weather matching complete',len(adjusted),'of',len(ss),flush=True)
if __name__=='__main__':run()
