"""Validate actual data provenance, analysis denominators, formulas and deliverable consistency.
This audit does not tune models or rerun statistical hypotheses.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
import hashlib,collections,math,ast,csv
import numpy as np
from study_io import *
from analysis import candidates,eligible

def run():
    inherited=read(RESULTS/'continuation_original_files.json');preservation=[]
    for name,info in inherited['files'].items():
        source=HERE/name
        if name.startswith('results/') and (RESULTS/'before_fix1').exists():
            source=RESULTS/'before_fix1'/name.removeprefix('results/')
        b=source.read_bytes()
        observed=b[:info['bytes']] if name in ['protocol.md','report.md'] else b
        preservation.append(dict(path=name,passed=hashlib.sha256(observed).hexdigest()==info['sha256']))
    assert all(x['passed'] for x in preservation)
    original=read(RAW/'MANIFEST.json');hashes=[]
    for item in original['items']:
        if 'sha256' not in item or 'path' not in item:continue
        f=RAW/item['path'];actual=hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
        hashes.append(dict(path=item['path'],passed=actual==item['sha256']))
    assert all(x['passed'] for x in hashes)
    whashes=[]
    for item in read(RAW/'weather/WEATHER_MANIFEST.json')['items']:
        if 'sha256' not in item or 'path' not in item:continue
        f=RAW/item['path']
        whashes.append(dict(path=item['path'],passed=f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']))
    assert all(x['passed'] for x in whashes)
    tc_cells=collections.defaultdict(set)
    for f in (RAW/'weather/terraclimate').glob('tc_*.json'):
        d=read(f)
        if d.get('valid_land'):tc_cells[tuple(f.stem.split('_')[1:3])].add((round(d['cell_lat'],6),round(d['cell_lon'],6)))
    assert all(len(v)==1 for v in tc_cells.values()), 'Overlapping TerraClimate files use different cells'
    data=read(RESULTS/'memory_regimes.json');rr=data['records'];ss={s['series_id']:s for s in candidates()};assert len(rr)==len({r['series_id'] for r in rr})
    assert len(rr)==read(RESULTS/'eligibility_audit.json')['primary_retained']
    for r in rr:
        assert r['n']>=15 and r['n_adjacent_pairs']>=12 and r['coverage']>=.75
        assert -.95<=r['r1_bc']<=.98
        assert r['sigma2']>0 and abs(r['sigma']**2-r['sigma2'])<1e-9
        assert r['long_D_method']=='direct_engine_refined'
        assert r['long_strict_failure']==(not r['long_iii_10pct'])
        assert r['short_penalty_zone_T2']==(r['r1_bc']>.1919)
    for name in ['10113','10114','10039','10040']:
        s=ss['GPDD_'+name];assert eligible(s)[0] is not None
    # No land site is assigned to marine on a Not-applicable Ocean token.
    gmain={r['MainID']:r for r in csv.DictReader((RAW/'gpdd/main.csv').open(encoding='latin1'))};loc={r['LocationID']:r for r in csv.DictReader((RAW/'gpdd/location.csv').open(encoding='latin1'))}
    examples=[]
    for mid in ['1','144','145','146','151']:
        if 'GPDD_'+mid in ss:examples.append(dict(series_id='GPDD_'+mid,realm=ss['GPDD_'+mid]['realm_detail'],Ocean=loc[gmain[mid]['LocationID']]['Ocean']))
    assert all(x['realm']!='Marine' for x in examples)
    weather=read(RESULTS/'weather_metrics.json');pm={r['series_id']:r for r in rr};matched=[w for w in weather if w['matched']]
    assert len(weather)==len(rr)
    assert {w['series_id'] for w in matched}=={w['series_id'] for w in read(RESULTS/'weather_adjusted_predictions.json.gz')}
    for w in matched:
        r=pm[w['series_id']];assert r['environment_type']!='laboratory'
        if r['realm_detail']=='Marine':assert w['source']=='NOAA_ERSST_v5'
        if w['source']=='TerraClimate_V1.1':assert w['distance_km']<=15
        if w['source'] in ['NOAA_ERSST_v5','NOAA_GHCNm_v4']:assert w['distance_km']<=250
        assert w['n_adjusted']>=15
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    history=RESULTS/'before_fix1' if (RESULTS/'before_fix1').exists() else RESULTS
    prior=read(history/'attempt3_before_metadata_correction/primary_metrics.json');oldids={r['series_id'] for r in prior}
    totals=dict(timestamp=now(),manifest_hashes_checked=len(hashes),manifest_hash_failures=sum(not x['passed'] for x in hashes),eligible_series=len(rr),study_clusters=len({r['study_id'] for r in rr}),added_after_metadata_correction=sorted({r['series_id'] for r in rr}-oldids),gpdd_realm_examples=examples,series_by_interval=dict(collections.Counter(r['interval_class'] for r in rr)),series_by_licence=dict(collections.Counter(r['licence'] for r in rr)),laboratory_records=[dict(series_id=r['series_id'],lab_regime=r['lab_regime']) for r in rr if r['environment_type']=='laboratory'],weather_matched=len(matched),bootstrap_counts={'within_series':200,'study_cluster':1000},negative_corrected_memory=sum(r['r1_bc']<0 for r in rr),corrected_memory_at_lower_bound=sum(r['r1_bc']<=-.95 for r in rr),corrected_memory_at_upper_bound=sum(r['r1_bc']>=.98 for r in rr),n_observations_quantiles=np.quantile([r['n'] for r in rr],[0,.5,1]).tolist(),calendar_year_range=[min(r['start_year'] for r in rr if r['start_year']),max(r['end_year'] for r in rr if r['end_year'])],numeric_extremes=[dict(series_id=r['series_id'],sigma=r['sigma']) for r in rr if r['sigma']>3],numerically_ambiguous={c:{k:sum(r[c+'_'+k+'_simulation_ambiguous'] for r in rr) for k in ['iii_10pct','iii_30pct']} for c in ['long','moderate','short']},hash_checks=hashes,passed=True)
    totals.update(inherited_file_preservation=preservation,weather_manifest_hashes_checked=len(whashes),weather_manifest_hash_checks=whashes,terraclimate_consistent_cell_groups=len(tc_cells))
    save(RESULTS/'final_audit.json',totals);print('AUDIT',json_text(totals),flush=True)
def json_text(x):
    import json
    return json.dumps(clean(x),indent=2)
if __name__=='__main__':run()
