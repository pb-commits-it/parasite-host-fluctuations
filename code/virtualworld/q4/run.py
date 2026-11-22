"""Execute one registered Q4 campaign; preserve completed outputs on failure."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();sys.path.insert(0,str(P))
from numerics import *
from world.simulation import simulate
from world.design import allocate
import platform,traceback,collections,scipy

def contrasts(records,cells):
    out=[];index={(r['series_id'],r['stages'],r['stay']):r for r in cells}
    groups=collections.defaultdict(list)
    for r in records:
        if r['source']=='Tribolium': groups[(r['dataset'],r['block'],r['stage'])].append(r)
    for (ds,block,stage),rr in sorted(groups.items()):
        control=[r for r in rr if r['treatment']==1]
        for treatment in sorted(set(r['treatment'] for r in rr)-{1}):
            treated=[r for r in rr if r['treatment']==treatment]
            for k in (2,3):
                for s in (1,5,20):
                    a=[index[r['series_id'],k,s] for r in treated];b=[index[r['series_id'],k,s] for r in control]
                    D=float(np.mean([r['D'] for r in a])-np.mean([r['D'] for r in b]))
                    item=dict(contrast_id=f'D{ds}_IC{block}_{stage}_T{treatment}_k{k}_s{s}',dataset=ds,block=block,
                        stage=stage,treatment=treatment,stages=k,stay=s,n_fluct=len(a),n_constant=len(b),D_contrast=D,
                        partial_source=bool(ds==8 and block==2 and stage=='A' and treatment==2),
                        members=';'.join(r['series_id'] for r in treated+control))
                    for name in ('AR','full'):
                        vals=[r[f'prediction_{name}'] for r in a+b]
                        pred=float(np.mean(vals[:len(a)])-np.mean(vals[len(a):])) if all(v is not None for v in vals) else None
                        item[f'prediction_{name}']=pred;item[f'H3_{name}']=status(D,pred)
                        item[f'tolerance_{name}']=tol(pred) if pred is not None else None
                    out.append(item)
    return out

if __name__=='__main__':
    if (P/'run-manifest.json').exists():raise RuntimeError('one campaign only; output exists')
    assert json.loads((P/'phase1-validation.json').read_text())['passed']
    assert json.loads((P/'numerical-validation.json').read_text())['passed']
    records=json.loads((P/'raw/selected-trajectories.json').read_text())
    start=time.monotonic()
    state=dict(started=date(),status='running',software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        hashes={f:digest(P/f) for f in ('protocol.md','numerics.py','prepare_data.py','run.py','selection.json')},
        count_records=len(records),batches=[],seconds=0.)
    def checkpoint():
        state['seconds']=time.monotonic()-start;write_json(P/'run-manifest.json',state)
        if state['seconds']>1800:raise RuntimeError('30 minute compute budget exhausted')
    checkpoint();cells=[];monotonic=[];diagnostics=[]
    try:
        # Every cell is saved immediately; failure never discards completed D.
        with (P/'cells.jsonl').open('x') as f:
            for i,r in enumerate(records):
                h=norm_host(r['values'])
                for k in (2,3):
                    previous=None
                    for s in (1,5,20):
                        D,residual=loss(k,s,h);ar,full=predict(k,s,h)
                        cell=dict(series_id=r['series_id'],series_index=i,source=r['source'],stages=k,stay=s,
                            D=D,root_residual=residual,prediction_AR=ar,prediction_full=full,H1_AR=status(D,ar),H1_full=status(D,full),
                            tolerance_AR=tol(ar) if ar is not None else None,tolerance_full=tol(full) if full is not None else None,
                            growth_at_logR0_005=growth(model(k,s),h,.05),invasion_at_logR0_005=bool(.05>D))
                        cells.append(cell);f.write(json.dumps(cell)+'\n');f.flush()
                        if previous is not None:
                            delta=previous['D']-D
                            monotonic.append(dict(series_id=r['series_id'],source=r['source'],stages=k,
                                from_stay=previous['stay'],to_stay=s,decrease=delta,
                                H2='decrease' if delta>1e-6 else ('reversal' if delta< -1e-6 else 'flat')))
                        previous=cell
                if (i+1)%24==0:print('thresholds',i+1,'/',len(records),'seconds',round(time.monotonic()-start,2),flush=True);checkpoint()
        write_csv(P/'cells.csv',cells);write_csv(P/'monotonicity.csv',monotonic)
        h3=contrasts(records,cells);write_csv(P/'contrasts.csv',h3)
        # Independent demographic draws, with uniform record phase; common phases across designs.
        phases={r['series_id']:np.random.default_rng(415150+i).integers(0,len(r['values']),64) for i,r in enumerate(records)}
        archives=P/'demography';archives.mkdir(exist_ok=True)
        survival=[]
        for k in (2,3):
            for si,s in enumerate((1,5,20)):
                m=model(k,s);T=sum(means(k,s));steps=max(500,100*T);initial=allocate(20,m.constant(.05)[1])
                for bi,left in enumerate(range(0,len(records),24)):
                    batch=records[left:left+24];nr=64*len(batch);seed=415200+1000*k+10*bi+si
                    hist=np.empty((steps,nr,1))
                    for j,r in enumerate(batch):
                        h=norm_host(r['values']);phase=phases[r['series_id']]
                        hist[:,64*j:64*(j+1),0]=h[(np.arange(steps)[:,None]+phase)%len(h)]
                    before=time.monotonic()
                    sim=simulate(m,np.tile(initial,(nr,1)),steps,seed,hosts=hist,y=.05)
                    ext=sim['extinction_time']
                    stem=f'k{k}_s{s}_batch{bi}'
                    np.savez_compressed(archives/(stem+'.npz'),extinction_time=ext,phases=np.concatenate([phases[r['series_id']] for r in batch]),
                        series_ids=np.array([r['series_id'] for r in batch]),initial=initial,horizon=steps,seed=seed,
                        censored=sim['censored'],final_total=sim['totals'][-1].astype(str))
                    for j,r in enumerate(batch):
                        for label,horizon in [('G23',23*T),('G50',50*T),('G100',100*T),('step500',500)]:
                            num=int(np.sum(ext[64*j:64*(j+1)]>horizon));lo,hi=wilson(num)
                            survival.append(dict(series_id=r['series_id'],source=r['source'],stages=k,stay=s,endpoint=label,
                                horizon=horizon,survivors=num,replicates=64,survival=num/64,ci_low=lo,ci_high=hi,
                                phase_seed=415150+left+j,demographic_seed=seed,batch=bi))
                    state['batches'].append(dict(stages=k,stay=s,batch=bi,series_ids=[r['series_id'] for r in batch],seed=seed,
                        seconds=time.monotonic()-before))
                    write_csv(P/'persistence.csv',survival);checkpoint()
                    print('demography',k,s,bi,'seconds',round(state['seconds'],2),flush=True)
                    del sim,hist
        trigger={c['series_id'] for c in cells if c['H1_AR']=='departure' or c['H1_full']=='departure'}
        trigger|={m['series_id'] for m in monotonic if m['H2']=='reversal'}
        for c in h3:
            if c['H3_AR']=='departure' or c['H3_full']=='departure':trigger.update(c['members'].split(';'))
        state['diagnostic_series']=sorted(trigger);checkpoint()
        with (P/'diagnostics.jsonl').open('x') as f:
            for i,r in enumerate(records):
                if r['series_id'] not in trigger:continue
                h=norm_host(r['values'])
                for kind,rep,v in variants(h,i):
                    for k in (2,3):
                        for s in (1,5,20):
                            D,res=loss(k,s,v);ar,full=predict(k,s,v)
                            row=dict(series_id=r['series_id'],stages=k,stay=s,variant=kind,replicate=rep,
                                D=D,prediction_AR=ar,prediction_full=full,root_residual=res)
                            diagnostics.append(row);f.write(json.dumps(row)+'\n')
                    f.flush()
                print('diagnostics',r['series_id'],'roots',len(diagnostics),'seconds',round(time.monotonic()-start,2),flush=True);checkpoint()
        write_csv(P/'diagnostics.csv',diagnostics)
        state.update(status='complete',finished=date(),primary_cells=len(cells),demographic_paths=len(records)*6*64,
            diagnostic_roots=len(diagnostics),diagnostic_count=len(trigger));checkpoint()
    except Exception:
        state.update(status='failed',finished=date(),exception=traceback.format_exc());checkpoint();raise
