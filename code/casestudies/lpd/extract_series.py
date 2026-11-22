"""Extract registered host trajectories and published location/date metadata.
Input is untrusted CSV/JSON only. Run with python3 -I. Does not estimate memory.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import csv, json, math, datetime as dt, hashlib, collections
import numpy as np
import helpers_continued as h
from study_io import *

def num(x):
    try:return float(x)
    except (ValueError,TypeError):return float('nan')
def rows(file,enc='latin1'):
    with file.open(encoding=enc) as f:return list(csv.DictReader(f))

def vecdyn():
    out=[];audit=collections.Counter();epoch=dt.date(1970,1,5)
    for jf in sorted((RAW/'vecdyn/full_datasets').glob('*.json')):
        d=read(jf);cons=d.get('consistent_data') or {};did=d['dataset_id']
        audit['files']+=1
        if 's41597-024-03482-y' in str(cons.get('curatedbydoi')) or 'temporally downscaled' in str(cons.get('additional_sample_info')).lower():
            audit['excluded_downscaled']+=1;continue
        events=collections.defaultdict(lambda:collections.defaultdict(set));ab=collections.defaultdict(lambda:collections.defaultdict(float));tax={};coords={};units={}
        lic=str(cons.get('dataset_license') or d.get('dataset_license') or 'CC-BY-NC-4.0')
        for i,r in enumerate(d.get('results') or []):
            def get(k,default=''):
                v=r.get(k);return v if v is not None and v!='' else cons.get(k,default)
            unit=str(get('sample_unit')).lower()
            if any(s in unit for s in ['presence','binary','percent']):continue
            try:dd=dt.date.fromisoformat(str(get('sample_end_date'))[:10]);v=float(get('sample_value'))
            except (ValueError,TypeError):continue
            if not math.isfinite(v) or v<0:continue
            la,lo=num(get('sample_lat_dd')),num(get('sample_long_dd'))
            place=str(get('sample_location') or get('location_description') or 'unspecified')
            loc=(round(la,3) if math.isfinite(la) else None,round(lo,3) if math.isfinite(lo) else None,place)
            key=(loc,str(get('sampling_method') or 'unspecified'),str(get('sample_stage') or 'unspecified'))
            coords[key]=(la,lo);units[key]=unit
            ev=str(get('sample_name') or f'{place}_{dd}')
            events[key][dd].add(ev)
            gen=str(get('genus'));sp=str(get('species'));fam=str(get('family')).lower()
            if sp.upper() in ['EMPTY','BLANK','NONE']:continue
            name=(gen+' '+sp).strip() if gen and not sp.lower().startswith(gen.lower()) else (sp or gen)
            tok=name.split()[0].lower() if name else ''
            if fam in h.TICK_FAMILIES or tok in h.TICK_GENERA:grp,sub='arachnid_vectors_ticks','ticks'
            elif fam in h.VECTOR_INSECT_FAMILIES or tok in h.VECTOR_INSECT_GENERA:grp,sub='insects_vectors','mosquitoes_and_biting_diptera'
            else:continue
            tax[name]=(grp,sub)
            ab[(key,name)][dd]+=v
        for key,em in events.items():
            dates=sorted(em)
            if len(dates)<15:continue
            gap=float(np.median(np.diff([(d-epoch).days for d in dates])))
            if 4<=gap<=10:iv,spy,bd='subannual_biweekly_weekly',52,7
            elif 11<=gap<=20:iv,spy,bd='subannual_biweekly_weekly',26,14
            elif 21<=gap<=45:iv,spy,bd='subannual_monthly',12,0
            elif 60<=gap<=130:iv,spy,bd='subannual_seasonal',4,0
            elif 300<=gap<=400:iv,spy,bd='annual',1,0
            else:audit['unregistered_interval']+=1;continue
            def bin_(d):
                return (d-epoch).days//bd if bd else d.year*spy+((d.month-1)//(12//spy) if spy>1 else 0)
            effort=collections.Counter()
            for dd,evs in em.items():effort[bin_(dd)]+=len(evs)
            bins=sorted(effort);la,lo=coords[key]
            for (key2,name),vals in ab.items():
                if key2!=key:continue
                sums=collections.Counter()
                for dd,v in vals.items():sums[bin_(dd)]+=v
                vv=np.array([sums[b]/effort[b] for b in bins])
                if (vv>0).sum()<15:continue
                sid=f'VD_{did}_{hashlib.sha256(repr((key,name)).encode()).hexdigest()[:12]}'
                dd=[epoch+dt.timedelta(days=b*bd) for b in bins] if bd else None
                out.append(dict(series_id=sid,database='VecDyn',study_id=f'VD_SET_{did}',licence=lic,cc_by_compatible=('nc' not in lic.lower()),taxon_group=tax[name][0],taxon_subgroup=tax[name][1],species=name,interval_class=iv,samples_per_year=float(spy),latitude=la,longitude=lo,realm='Terrestrial',realm_detail='Terrestrial',environment_type='field',reliability=4,manipulated=False,series_kind='primary',steps=np.array(bins),values=vv,subperiod=np.array([b%spy for b in bins]),step_years=np.array([d.year for d in dd]) if dd else np.array([b//spy for b in bins]),step_months=np.array([d.month for d in dd]) if dd else np.array([(b%spy)*(12//spy)+1 if spy>1 else 0 for b in bins]),step_dates_iso=[d.isoformat() for d in dd] if dd else None,bin_days=bd,published_location=key[0][2],location_uncertainty='Coordinates rounded to 0.001 degree for grouping; original spatial accuracy unspecified',units=units[key],lab_regime='not applicable',date_precision='bin of recorded sample end dates'))
    audit['candidate_series']=len(out);return out,dict(audit)

def extract():
    ss,audit=h.load_all_raw_series(RAW)
    vd,va=vecdyn();ss.extend(vd);audit['vecdyn_registered_location_loader']=va
    gm={r['MainID']:r for r in rows(RAW/'gpdd/main.csv')};gl={r['LocationID']:r for r in rows(RAW/'gpdd/location.csv')};gb={r['BiotopeID']:r for r in rows(RAW/'gpdd/biotope.csv')}
    bm={r['STUDY_ID']:r for r in rows(RAW/'biotime/BioTIMEMetadata_24_06_2021.csv')}
    for s in ss:
        if s['database']=='GPDD':
            m=gm[s['series_id'].split('_')[1]];l=gl.get(m.get('LocationID'),{});b=gb.get(m.get('BiotopeID'),{})
            s['published_location']='; '.join(l.get(k,'') for k in ['ExactName','TownName','CountyStateProvince','Country'])
            s['location_uncertainty']='SpatialAccuracy='+l.get('SpatialAccuracy','')+'; LocationExtent='+l.get('LocationExtent','')+'; bounds='+repr([l.get(k) for k in ['North','East','South','West']])
            s['units']=m.get('SamplingUnits','unspecified');txt=' '.join([m.get('Notes',''),b.get('HabitatName',''),b.get('BiotopeType','')])
            if any(w in txt.lower() for w in ['laboratory experiment','laboratory population','microcosm','aquarium','indoor','laboratory culture']):s['environment_type']='laboratory'
            s['lab_regime']=txt if s['environment_type']=='laboratory' else 'not applicable'
            s['date_precision']='published SampleYear and TimePeriodID; representative month for seasons'
        elif s['database']=='BioTIME':
            st=bm[s['study_id'].replace('BT_STUDY_','')]
            s['published_location']=st.get('TITLE','')+'; '+st.get('HABITAT','')
            s['location_uncertainty']='Study centroid; AREA_SQ_KM='+st.get('AREA_SQ_KM','')+'; NUMBER_LAT_LONG='+st.get('NUMBER_LAT_LONG','')+'; within-study location unspecified here'
            s['units']=st.get('ABUNDANCE_TYPE','unspecified')+' per sampled event'
            txt=' '.join(st.get(k,'') for k in ['HABITAT','METHODS','COMMENTS'])
            if any(w in txt.lower() for w in ['laboratory microcosm','laboratory culture','indoor aquarium']):s['environment_type']='laboratory'
            s['lab_regime']=txt if s['environment_type']=='laboratory' else 'not applicable'
            s['date_precision']='calendar aggregation; centroid used for pooled assemblage'
            if s['series_kind']!='primary':
                spy=int(s['samples_per_year']);s['step_years']=s['steps']//spy;s['step_months']=np.array([(int(b)%spy)*(12//spy)+1 if spy>1 else 0 for b in s['steps']]);s['subperiod']=s['steps']%spy
        if s['database']=='BioTIME':s['cc_by_compatible']=s['cc_by_compatible'] and 'nc' not in s['licence'].lower()
    save(RAW/'derived/candidate_series.json.gz',ss);save(RESULTS/'extraction_audit.json',dict(timestamp=now(),counts=audit,total=len(ss)))
    print(json.dumps(dict(timestamp=now(),counts=audit,total=len(ss)),indent=2),flush=True)

if __name__=='__main__':extract()
