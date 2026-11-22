"""Apply frozen inclusion, stratification and selection before any outcomes."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();sys.path.insert(0,str(P))
from numerics import *
import gzip,collections,openpyxl

if __name__=='__main__':
    if (P/'selection.json').exists(): raise RuntimeError('append-only selection')
    assert json.loads((P/'phase1-validation.json').read_text())['passed']
    raw=ROOT/'data/raw/lpd/derived/candidate_series.json.gz'
    metap=ROOT/'casestudies/lpd/results/series_metrics.csv'
    allraw=json.load(gzip.open(raw,'rt'));byid={r['series_id']:r for r in allraw}
    assert len(byid)==len(allraw)
    metrics=list(csv.DictReader(metap.open()));assert len(set(r['series_id'] for r in metrics))==len(metrics)
    groups=collections.defaultdict(list);excluded=[]
    for meta in metrics:
        sid=meta['series_id'];r=byid.get(sid);reason=''
        if r is None: reason='missing_raw'
        else:
            try: vals=np.asarray(r['values'],float);steps=np.asarray(r['steps'],float)
            except (ValueError,TypeError): reason='nonnumeric'
            if not reason:
                if len(vals)<16: reason='short'
                elif len(vals)!=len(steps) or np.any(~np.isfinite(vals)) or np.any(vals<0) or np.any(~np.isfinite(steps)): reason='invalid'
                elif vals.mean()<=0: reason='zero_mean'
                elif not np.all(np.diff(steps)==1): reason='gap_or_irregular'
                elif not r.get('cc_by_compatible'): reason='licence'
        if reason: excluded.append(dict(series_id=sid,reason=reason));continue
        key=(meta['taxon_group'],meta['interval_class'])
        groups[key].append(sid)
    keys=sorted(groups);cap=np.array([len(groups[k]) for k in keys]);target=min(120,int(cap.sum()))
    alloc=np.ones(len(keys),int)
    remaining=target-len(keys)
    quota=remaining*(cap-1)/(cap-1).sum();extra=np.floor(quota).astype(int);alloc+=extra
    for j in sorted(range(len(keys)),key=lambda j:(-(quota[j]-extra[j]),keys[j]))[:target-int(alloc.sum())]: alloc[j]+=1
    rng=np.random.default_rng(415111);ids=[];strata=[]
    for key,num,capacity in zip(keys,alloc,cap):
        picked=sorted(rng.choice(sorted(groups[key]),int(num),replace=False).tolist());ids.extend(picked)
        strata.append(dict(taxon_group=key[0],interval_class=key[1],eligible=int(capacity),selected=int(num)))
    ids=sorted(ids);selected=[]
    for sid in ids:
        r=byid[sid]
        selected.append(dict(series_id=sid,source=r['database'],study_id=r['study_id'],taxon_group=r['taxon_group'],
            interval_class=r['interval_class'],samples_per_year=r['samples_per_year'],licence=r['licence'],
            species=r['species'],values=r['values'],steps=r['steps'],dataset='',treatment='',block='',stage='',replicate=''))
    write_csv(P/'selected-supplied-metrics.csv',[r for r in metrics if r['series_id'] in set(ids)])
    write_csv(P/'excluded-compilation.csv',excluded);write_csv(P/'strata.csv',strata)
    workbook=P/'raw/7972066/BeetleTeamTriboliumDataArchive.xlsx'
    w=openpyxl.load_workbook(workbook,read_only=True,data_only=True)
    trib=[];excluded_trib=[]
    for ds in (7,8,10):
        rows=list(w[f'Data Set {ds}'].values);head=rows[0]
        grouped=collections.defaultdict(list)
        for row in rows[1:]:
            r=dict(zip(head,row))
            if ds in (7,10) and r['Week']<18:continue
            key=(r.get('Treatment',r.get('HS')),r.get('IC',0),r['Replicate'])
            grouped[key].append(r)
        for (treat,block,rep),rs in sorted(grouped.items()):
            rs=sorted(rs,key=lambda r:r['Week']);weeks=[r['Week'] for r in rs]
            assert len(rs)>=16 and np.all(np.diff(weeks)==2)
            for stage in ('L','A'):
                vals=[r[f'{stage}-stage'] for r in rs]
                sid=f'TRIB_D{ds}_T{treat}_IC{block}_R{rep}_{stage}'
                if np.any(np.asarray(vals)<0) or not np.all(np.isfinite(vals)) or np.mean(vals)<=0:
                    excluded_trib.append(dict(series_id=sid,reason='invalid_count',
                        invalid_weeks=';'.join(str(r['Week']) for r in rs if r[f'{stage}-stage'] is None or r[f'{stage}-stage']<0)))
                    continue
                trib.append(dict(series_id=sid,source='Tribolium',study_id=f'Tribolium_D{ds}',
                    taxon_group='laboratory_insects',interval_class='biweekly',samples_per_year=26,
                    licence='CC0-1.0',species='Tribolium castaneum',values=vals,
                    steps=[(v-weeks[0])/2 for v in weeks],weeks=weeks,habitat=[r['Habitat'] for r in rs],
                    dataset=ds,treatment=treat,block=block,replicate=rep,stage=stage))
    assert len(trib)<=72
    write_csv(P/'excluded-tribolium.csv',excluded_trib)
    selected+=trib
    fingerprints=collections.defaultdict(list)
    for r in selected:
        fingerprint=hashlib.sha256(np.asarray(r['values'],dtype='<f8').tobytes()).hexdigest()
        fingerprints[fingerprint].append(r['series_id'])
    overlaps=[v for v in fingerprints.values() if len(v)>1]
    for i,r in enumerate(selected):r['series_index']=i
    # Raw trajectories are intentionally kept in ignored storage only.
    write_json(P/'raw/selected-trajectories.json',selected)
    records=[]
    for r in selected:
        rec={k:v for k,v in r.items() if k not in ('values','steps','weeks','habitat')}
        h=norm_host(r['values']);rec.update(descriptors(h))
        rec['observed_mean']=float(np.mean(r['values']));rec['normalized_mean']=float(h.mean())
        rec['observed_habitat_mean']=float(np.mean(r['habitat'])) if 'habitat' in r else None
        rec['first_time']=r.get('weeks',r['steps'])[0];rec['last_time']=r.get('weeks',r['steps'])[-1]
        rec['trajectory_sha256']=hashlib.sha256(np.asarray(r['values'],dtype='<f8').tobytes()).hexdigest()
        records.append(rec)
    write_csv(P/'series.csv',records)
    write_json(P/'selection.json',dict(date=date(),seed=415111,eligible=int(cap.sum()),excluded=len(excluded),
        compilation_count=len(ids),tribolium_count=len(trib),ids=[r['series_id'] for r in selected],
        overlaps=overlaps,strata=strata,input_hashes={str(p.relative_to(ROOT)):digest(p) for p in (raw,metap,workbook)},
        protocol_sha256=digest(P/'protocol.md'),code_sha256=digest(P/'prepare_data.py')))
    print('selected',len(ids),'compilation +',len(trib),'Tribolium; eligible',cap.sum(),'overlap groups',len(overlaps))
