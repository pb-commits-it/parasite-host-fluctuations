"""Complete registered Q4 PGF persistence and diagnostics; no repeated D run."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();sys.path.insert(0,str(P))
from numerics import *
from pgf import extinction
from world.design import allocate
import traceback

if __name__=='__main__':
    dest=P/'completion-manifest.json'
    if dest.exists():raise RuntimeError('append-only completion')
    assert json.loads((P/'pgf-validation.json').read_text())['passed']
    records=json.loads((P/'raw/selected-trajectories.json').read_text())
    cells=[json.loads(line) for line in (P/'cells.jsonl').read_text().splitlines()]
    mono=list(csv.DictReader((P/'monotonicity.csv').open()));contr=list(csv.DictReader((P/'contrasts.csv').open()))
    assert len(cells)==len(records)*6
    trigger={r['series_id'] for r in cells if r['H1_AR']=='departure' or r['H1_full']=='departure'}
    trigger|={r['series_id'] for r in mono if r['H2']=='reversal'}
    for r in contr:
        if r['H3_AR']=='departure' or r['H3_full']=='departure':trigger.update(r['members'].split(';'))
    start=time.monotonic();state=dict(started=date(),status='running',interrupted_run_seconds=128.,
        interrupted_run_note='KeyboardInterrupt in exact binomial chunks; original manifest is an interrupted checkpoint, not completion',
        input_hashes={name:digest(P/name) for name in ('cells.jsonl','selection.json','protocol.md','pgf.py','complete.py','numerics.py')},
        diagnostic_series=sorted(trigger))
    def checkpoint():
        state['seconds']=time.monotonic()-start;state['total_campaign_seconds']=state['seconds']+state['interrupted_run_seconds']
        write_json(dest,state)
        if state['total_campaign_seconds']>1800:raise RuntimeError('compute budget')
    checkpoint();survival=[];phase_output={};diagnostics=[]
    try:
        phase_dir=P/'phase-persistence';phase_dir.mkdir(exist_ok=True)
        for i,r in enumerate(records):
            h=norm_host(r['values']);data={}
            for k in (2,3):
                for s in (1,5,20):
                    m=model(k,s);T=sum(means(k,s));init=allocate(20,m.constant(.05)[1])
                    endpoints={'G23':23*T,'G50':50*T,'G100':100*T,'step500':500}
                    out=extinction(m,h,init,list(endpoints.values()))
                    for label,t in endpoints.items():
                        p=out[t];data[f'k{k}_s{s}_{label}']=p
                        if np.any(~np.isfinite(p)) or min(p)<-1e-12 or max(p)>1+1e-12:raise RuntimeError('invalid persistence')
                        survival.append(dict(series_id=r['series_id'],source=r['source'],stages=k,stay=s,endpoint=label,horizon=t,
                            survival=float(p.mean()),phase_min=float(p.min()),phase_max=float(p.max()),phases=len(h),
                            initial=';'.join(map(str,init)),method='exact_phase_averaged_branching_PGF'))
            np.savez_compressed(phase_dir/(r['series_id']+'.npz'),**data)
            if (i+1)%24==0:
                write_csv(P/'persistence-exact.csv',survival);checkpoint()
                print('PGF',i+1,'/',len(records),'seconds',round(state['seconds'],2),flush=True)
        write_csv(P/'persistence-exact.csv',survival)
        # Compare every preserved count endpoint with its exact conditional mean.
        comparison=[]
        for r in csv.DictReader((P/'persistence.csv').open()):
            phase=np.random.default_rng(int(r['phase_seed'])).integers(0,len(next(x['values'] for x in records if x['series_id']==r['series_id'])),64)
            with np.load(phase_dir/(r['series_id']+'.npz')) as data:p=data[f"k{r['stages']}_s{r['stay']}_{r['endpoint']}"][phase]
            expect=float(p.mean());err=abs(float(r['survival'])-expect);bound=4*np.sqrt(np.sum(p*(1-p)))/64+.002
            comparison.append(dict(series_id=r['series_id'],stages=int(r['stages']),stay=int(r['stay']),endpoint=r['endpoint'],
                observed=float(r['survival']),expected=expect,error=err,tolerance=float(bound),passed=bool(err<=bound)))
        write_csv(P/'demographic-pgf-checks.csv',comparison)
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
        state.update(status='complete',finished=date(),persistence_cells=len(survival),diagnostic_roots=len(diagnostics),
            preserved_MC_checks=len(comparison),preserved_MC_checks_passed=sum(r['passed'] for r in comparison))
        checkpoint()
    except BaseException:
        state.update(status='failed',finished=date(),exception=traceback.format_exc());checkpoint();raise
