"""List original sampling-event coordinates/dates used directly or in effort denominators.
No abundance values are copied. Fine source-event ledger remains under raw/derived.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
import csv,gzip,io,zipfile,collections,math
from study_io import *
from analysis import candidates,eligible

def run():
    ss=candidates();used=[]
    for s in ss:
        ok=any(eligible(s,mode)[0] is not None for mode in ['primary','S1_zero','S7_coverage50']) if s['series_kind']=='primary' else eligible(s)[0] is not None and not s['manipulated']
        if ok:used.append(s)
    bystudy=collections.defaultdict(list)
    for s in used:bystudy[s['study_id']].append(s)
    dest=RAW/'derived/source_field_record_locations.csv.gz';counts=collections.Counter();bounds={}
    fields=['database','study_id','source_record_id','series_id','year','month','day','date_end','latitude','longitude','location_note','environment_type']
    with gzip.open(dest,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        def emit(r):
            writer.writerow(r);counts[r['database']]+=1
            try:
                la,lo=float(r.get('latitude')),float(r.get('longitude'))
                if not(-90<=la<=90 and -180<=lo<=180):return
                b=bounds.setdefault(r['study_id'],[la,la,lo,lo]);b[0]=min(b[0],la);b[1]=max(b[1],la);b[2]=min(b[2],lo);b[3]=max(b[3],lo)
            except (TypeError,ValueError):pass
        mids={s['series_id']:s for s in used if s['database']=='GPDD'}
        with (RAW/'gpdd/data.csv').open(encoding='latin1') as g:
            for r in csv.DictReader(g):
                s=mids.get('GPDD_'+r['MainID'])
                if s is None:continue
                emit(dict(database='GPDD',study_id=s['study_id'],source_record_id=r['DataID'],series_id=s['series_id'],year=r['SampleYear'],month=r['TimePeriodID'],latitude=s['latitude'],longitude=s['longitude'],location_note='month column is original TimePeriodID; see timeperiod.csv; '+s['location_uncertainty'],environment_type=s['environment_type']))
        with zipfile.ZipFile(RAW/'biotime/BioTIMEQuery_24_06_2021.zip') as z:
            with z.open('BioTIMEQuery_24_06_2021.csv') as b:
                rd=csv.DictReader(io.TextIOWrapper(b,encoding='latin1'))
                for i,r in enumerate(rd,2):
                    sid='BT_STUDY_'+r['STUDY_ID'];lst=bystudy.get(sid)
                    if not lst:continue
                    # Every assemblage row contributes to effort-event discovery, including non-target taxa.
                    emit(dict(database='BioTIME',study_id=sid,source_record_id='csv_line_'+str(i),year=r['YEAR'],month=r['MONTH'],day=r['DAY'],latitude=r['LATITUDE'],longitude=r['LONGITUDE'],location_note='source sample '+r['SAMPLE_DESC']+'; plot '+r['PLOT'],environment_type=lst[0]['environment_type']))
        for jf in sorted((RAW/'vecdyn/full_datasets').glob('*.json')):
            d=read(jf);sid='VD_SET_'+str(d['dataset_id']);lst=bystudy.get(sid)
            if not lst:continue
            cons=d.get('consistent_data') or {}
            for i,r in enumerate(d.get('results') or []):
                def val(k):return r.get(k) if r.get(k) is not None else cons.get(k,'')
                emit(dict(database='VecDyn',study_id=sid,source_record_id=str(i),date_end=val('sample_end_date'),latitude=val('sample_lat_dd'),longitude=val('sample_long_dd'),location_note='source location; full sampled dataset to preserve effort',environment_type='field'))
    # Aggregated-bin ledger covers every primary/sensitivity trajectory, including zero/missing status.
    dest2=RAW/'derived/analysis_bin_locations.csv.gz'
    with gzip.open(dest2,'wt',newline='') as f:
        w=csv.writer(f);w.writerow(['series_id','study_id','step','year','month','date_start','latitude','longitude','environment_type','positive_for_primary','date_precision'])
        for s in used:
            for i,t in enumerate(s['steps']):w.writerow([s['series_id'],s['study_id'],t,s['step_years'][i],s['step_months'][i],s['step_dates_iso'][i] if s['step_dates_iso'] else '',s['latitude'],s['longitude'],s['environment_type'],bool(s['values'][i]>0),s['date_precision']])
    out=dict(generated_utc=now(),source_record_counts=dict(counts),series_including_sensitivities=len(used),study_coordinate_bounds=bounds,source_ledger=str(dest.relative_to(ROOT)),bin_ledger=str(dest2.relative_to(ROOT)),warning='Source ledger includes the complete sampled assemblage/dataset for effort accounting. GPDD month is TimePeriodID. Weather for pooled BioTIME uses registered study centroids, not each original sample location; bounds expose that mismatch.')
    save(RESULTS/'record_inventory.json',out);print(out['source_record_counts'],out['series_including_sensitivities'],flush=True)
if __name__=='__main__':run()
