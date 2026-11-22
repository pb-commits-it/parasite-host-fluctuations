"""Append-only fix1 validation and the two prespecified sensitivity runs.

Run with -I, then one of: validate, pilot, matched, verify.
Validation hashes the corrected implementation and registered protocols;
each scientific run checks those hashes and refuses an existing destination.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0,str(HERE))
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
import numpy as np
import scipy
from world import hosts
from world.model import ring
from world.simulation import simulate, extinct_by
from world.counts import archive
from world.design import allocate, GENERATIONS, generation_steps, matched_run_steps
from run_pilot import contrast, date, wilson, main as pilot

OUT=HERE/'results/fix1'
REGISTRATION=(__import__('pathlib').Path(__file__).resolve().parent / '../protocols/vworld-fix1.md').resolve()
MARKER=b'\n## 2026-11-15 addendum'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    paths=sorted((HERE/'world').glob('*.py'))+sorted((HERE/'tests').glob('test_*.py'))
    paths += [HERE/'run_pilot.py',HERE/'run_fix1.py',HERE/'protocol.md',REGISTRATION]
    return {str(p.relative_to(HERE.parent)):digest(p) for p in paths}


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def validate(suffix=""):
    OUT.mkdir(parents=True,exist_ok=True)
    log=OUT/f'tests{suffix}.log'; record=OUT/f'validation{suffix}.json'
    if log.exists() or record.exists():raise RuntimeError('fix1 validation already exists')
    start=time.monotonic(); frozen=inputs()
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-I','-m','unittest','discover','-s',str(HERE/'tests'),
                               '-p','test_*.py','-v'],stdout=stream,stderr=subprocess.STDOUT,timeout=1500)
    record_data=dict(status='PASS' if result.returncode==0 else 'FAIL',completed=date(),
                     seconds=time.monotonic()-start,inputs=frozen,
                     python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    write_json(record,record_data)
    if result.returncode:raise RuntimeError('fix1 checks failed; scientific runs prohibited')
    print('PASS: complete suite and known-law gates',flush=True)


def preflight():
    record=OUT/'validation-correction1.json'
    if not record.exists():record=OUT/'validation.json'
    checks=json.loads(record.read_text())
    if checks['status']!='PASS' or checks['inputs']!=inputs():
        raise RuntimeError('passing checks on the current registered implementation required')
    original=json.loads((HERE/'results/validation.json').read_text())
    frozen=(HERE/'protocol.md').read_bytes().split(MARKER)[0]
    frozen_hash=hashlib.sha256(frozen).hexdigest()
    if original['status']!='PASS' or frozen_hash!=original['protocol_sha256']:
        raise RuntimeError('frozen phase-1 protocol and passing original gates required')
    return frozen_hash


def run_pilot():
    original=preflight()
    path=OUT/'pilot'
    if path.exists():raise RuntimeError('pilot sensitivity destination already exists')
    pilot(results=path,report_path=HERE/'pilot-rerun-fix1.md',validation_digest=original,sensitivity=True)


def run_matched():
    preflight()
    dest=OUT/'matched'; report=HERE/'matched-generations-fix1.md'
    if dest.exists() or report.exists():raise RuntimeError('matched run already exists')
    dest.mkdir(); start=time.monotonic()
    metadata=dict(started=date(),inputs=inputs(),n=400,generations=GENERATIONS,
                  phi=.6,log_R0=.05,status='running')
    write_json(dest/'run.json',metadata)
    rows={}; by_horizon={str(g):{} for g in GENERATIONS};by_horizon['steps500']={}
    for sigma in (0.,.6):
        for stay in (1,20):
            idx=len(rows);seed=962000+10*idx; key=f's{sigma:g}_t{stay}'
            model=ring([2,stay]);init=allocate(20,model.constant(.05)[1]);steps=matched_run_steps(stay)
            h=hosts.lognormal(steps,400,sigma,.6,seed)
            result=simulate(model,np.tile(init,(400,1)),steps,seed+1,hosts=h,y=.05)
            np.savez_compressed(dest/(key+'.npz'),totals=archive(result['totals']),
                                extinction_time=result['extinction_time'],censored=result['censored'],
                                initial=init,hosts=h)
            row=dict(cell=key,sigma=sigma,stay=stay,seed=seed,demographic_seed=seed+1,
                     steps=steps,initial=init.tolist(),n=400)
            for horizon in by_horizon:
                t=500 if horizon=='steps500' else generation_steps(stay,int(horizon))
                k=extinct_by(result,t);lo,hi=wilson(k,400)
                by_horizon[horizon][key]=dict(**row,horizon_steps=t,extinct=k,
                                              extinction=k/400,ci95=[lo,hi])
            rows[key]=row
            write_json(dest/(key+'.json'),{g:v[key] for g,v in by_horizon.items()})
            print(key,'extinctions',{g:v[key]['extinct'] for g,v in by_horizon.items()},flush=True)
            if time.monotonic()-start>600:raise RuntimeError('matched run budget exceeded')
    effects={}
    keys=['s0.6_t1','s0_t1','s0.6_t20','s0_t20']
    for g,table in by_horizon.items():
        effects[g]=contrast(table,{},keys,[1,-1,-1,1],'extinction',seed=963000+(500 if g=='steps500' else int(g)))
    allocation=max(effects[str(g)]['n80_per_cell'] for g in GENERATIONS)
    write_json(dest/'cells.json',by_horizon)
    write_json(dest/'effects.json',effects)
    metadata.update(completed=date(),seconds=time.monotonic()-start,status='complete',
                    planning_n80_per_cell=allocation)
    write_json(dest/'run.json',metadata)
    lines=['# Matched-generation sensitivity: vworld-fix1 — 2026-11-15','',
           f"Started {metadata['started']}; completed {metadata['completed']}. Runtime {metadata['seconds']:.2f} seconds.",
           '', 'Numerical result: four independent cells, 400 trajectories/cell (1,600 total). '
           'Exposed mean 2; steady mean 1 or 20; sigma 0 or .6; phi=.6; ln R0=.05; '
           'initial 20 allocated [13,7] or [2,18]. Seeds 962000, 962010, 962020, 962030 '
           'for hosts in sigma-then-stay order; demographic seed is host seed+1. '
           'Each trajectory covers G=100 and step 500; horizons share trajectories. No protocol departures.',
           '', 'T_gen=2+stay is a reference circuit time, not a realized lineage count. '
           'All four-cell interactions use independent cells. Comparisons between G values '
           'are correlated and are not tested as independent. G=50 was registered as the '
           'single confirmatory horizon; the three primary time-course estimates have pointwise intervals.',
           '', '| Horizon | Steps (stay 1 / 20) | p(0,1) | p(.6,1) | p(0,20) | p(.6,20) | Interaction [95% Wald] | n/cell, 80% | Planning 10–90% |',
           '|---|---|---:|---:|---:|---:|---|---:|---|']
    for g in [str(x) for x in GENERATIONS]+['steps500']:
        table=by_horizon[g];c=effects[g]
        ps=[table[k]['extinction'] for k in ('s0_t1','s0.6_t1','s0_t20','s0.6_t20')]
        times='500 / 500' if g=='steps500' else f'{generation_steps(1,int(g))} / {generation_steps(20,int(g))}'
        label='500 steps (secondary)' if g=='steps500' else f'G={g}'
        lines.append(f"| {label} | {times} | "+' | '.join(f'{p:.4f}' for p in ps)+
                     f" | {c['effect']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] | {c['n80_per_cell']} | "
                     f"{c['n80_sensitivity_10_90'][0]:.0f}–{c['n80_sensitivity_10_90'][1]:.0f} |")
    lines += ['', f'Planning allocation: **{allocation} replicates per cell**, the maximum of the three matched-horizon '
              'point estimates, for marginal 80% power. This is not an estimate of joint power.',
              '', 'Equal-allocation normal planning uses two-sided alpha=.05 and '
              'n=ceil((z_.975+z_.8)^2 * sum p_i(1-p_i) / delta^2). '
              'These are model-conditional planning estimates, not measured achieved power or real-trial sample sizes.',
              '', 'Independent Jeffreys-beta parameter sensitivity (100,000 interaction draws; '
              'first 10,000 for power; seeds 963000+G) gives:']
    for g in map(str,GENERATIONS):
        c=effects[g];ci=c['jeffreys_interval95']
        lines += [f"- G={g}: interaction [{ci[0]:.4f}, {ci[1]:.4f}]. "+
                  ('The interval excludes zero.' if c['finite_upper_established'] else
                   'The interval includes zero; a finite upper replication requirement is not established.')]
    lines += ['', 'Per-cell Wilson intervals, counts, seeds, first extinctions, censoring, full histories '
              'and host paths are in results/fix1/matched/. Inputs and software validation hashes are archived. '
              'All requested cells and horizons completed; this run is closed.']
    report.write_text('\n'.join(lines)+'\n')


def complete_matched():
    preflight()
    dest=OUT/'matched-secondary-completion'
    report=HERE/'matched-secondary-completion-fix1.md'
    if dest.exists() or report.exists():raise RuntimeError('matched completion already exists')
    dest.mkdir(); start=time.monotonic()
    table=json.loads((OUT/'matched/cells.json').read_text())['steps500']
    replay=[]
    for key in ('s0_t1','s0.6_t1'):
        row=table[key];model=ring([2,1]);seed=row['seed']
        h=hosts.lognormal(500,400,row['sigma'],.6,seed)
        sim=simulate(model,np.tile(row['initial'],(400,1)),500,seed+1,hosts=h,y=.05)
        with np.load(OUT/'matched'/f'{key}.npz',allow_pickle=False) as old:
            same_hosts=np.array_equal(old['hosts'],h[:300])
            same_totals=np.array_equal(old['totals'],sim['totals'][:301])
            if not (same_hosts and same_totals):raise RuntimeError('seeded replay disagrees; stop')
        replay.append(dict(cell=key,hosts_identical=same_hosts,totals_identical=same_totals))
        k=extinct_by(sim,500);lo,hi=wilson(k,400)
        row.update(steps=500,extinct=k,extinction=k/400,ci95=[lo,hi])
        np.savez_compressed(dest/f'{key}.npz',totals=archive(sim['totals']),
                            extinction_time=sim['extinction_time'],censored=sim['censored'],
                            initial=row['initial'],hosts=h)
        if time.monotonic()-start>600:raise RuntimeError('completion compute budget exceeded')
    keys=['s0.6_t1','s0_t1','s0.6_t20','s0_t20']
    effect=contrast(table,{},keys,[1,-1,-1,1],'extinction',seed=963500)
    write_json(dest/'cells.json',table);write_json(dest/'effect.json',effect)
    meta=dict(completed=date(),seconds=time.monotonic()-start,status='complete',replay_checks=replay,
              inputs=inputs(),additional_independent_replicates=0)
    write_json(dest/'run.json',meta)
    lines=['# Secondary-horizon completion: vworld-fix1 — 2026-11-15','',
           'This corrects the withdrawn 500-step row in matched-generations-fix1.md. '
           'The two stay-one cells were replayed to step 500 using the original seeds. '
           'All host values in their first 300 steps and all count totals through step '
           '300 match bit for bit. No independent replicates were added. '
           'All primary G=23,50,100 estimates and power results remain unchanged.',
           '', '| sigma | stay | extinct/400 by step 500 | probability [95% Wilson] |',
           '|---:|---:|---:|---|']
    for row in table.values():
        lines.append(f"| {row['sigma']} | {row['stay']} | {row['extinct']} | "
                     f"{row['extinction']:.4f} [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] |")
    lines += ['', f"Corrected secondary interaction: **{effect['effect']:.4f} "
              f"[{effect['ci95'][0]:.4f}, {effect['ci95'][1]:.4f}]** (95% Wald). "
              f"Normal planning n80={effect['n80_per_cell']}/cell applies only to this secondary outcome. "
              'Phase-2 primary planning remains 58, 61 and 51/cell at G=23,50,100, respectively; '
              'maximum 61/cell for marginal 80% power.',
              '', f"Completed {meta['completed']}; measured completion compute {meta['seconds']:.2f} seconds. "
              'Departure and replay criteria were registered before completion. '
              'Artifacts: results/fix1/matched-secondary-completion/. '
              'The horizon guard and full 22-test suite passed before replay. This completion is closed.']
    report.write_text('\n'.join(lines)+'\n')
    print('PASS: identical replay; corrected secondary',effect['effect'],effect['ci95'])


def verify():
    """Read-only outcome accounting; this does not simulate more worlds."""
    checks=[]
    def check(label,ok):
        checks.append(dict(check=label,passed=bool(ok)))
    originals=json.loads((HERE/'fix1-original-hashes.json').read_text())
    for name,expected in originals.items():
        raw=(HERE.parents[2]/name).read_bytes()
        if name.endswith('/protocol.md') or name.endswith('/README.md'):
            raw=raw.split(MARKER)[0]
        check('preserved '+name,hashlib.sha256(raw).hexdigest()==expected)
    table=list(csv.DictReader((OUT/'pilot/pilot_cells.csv').open()))
    check('18 pilot cells',len(table)==18)
    for row in table:
        with np.load(OUT/'pilot'/f"{row['cell']}.npz",allow_pickle=False) as d:
            zero=d['totals']==0
            first=np.where(zero.any(axis=0),zero.argmax(axis=0),501)
            check(row['cell']+' first extinctions',np.array_equal(first,d['extinction_time']))
            check(row['cell']+' reported count',np.sum(first<=500)==int(row['extinct']))
            check(row['cell']+' initial total',np.all(d['totals'][0]==20))
    tables=json.loads((OUT/'matched/cells.json').read_text())
    effects=json.loads((OUT/'matched/effects.json').read_text())
    completion=OUT/'matched-secondary-completion'
    if completion.exists():
        tables['steps500']=json.loads((completion/'cells.json').read_text())
        effects['steps500']=json.loads((completion/'effect.json').read_text())
    for horizon,table in tables.items():
        ps=[]
        for key in ('s0.6_t1','s0_t1','s0.6_t20','s0_t20'):
            row=table[key]
            folder=completion if horizon=='steps500' and row['stay']==1 and completion.exists() else OUT/'matched'
            with np.load(folder/f'{key}.npz',allow_pickle=False) as d:
                zero=d['totals']==0
                first=np.where(zero.any(axis=0),zero.argmax(axis=0),row['steps']+1)
                count=int(np.sum(first<=row['horizon_steps']))
                check(horizon+key+' accounting',count==row['extinct'] and np.array_equal(first,d['extinction_time']))
                ps.append(count/len(first))
        delta=ps[0]-ps[1]-ps[2]+ps[3]
        var=sum(p*(1-p) for p in ps)
        # Independent normal quantiles from stdlib, not the production helper.
        from statistics import NormalDist
        factor=(NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.8))**2
        check(horizon+' effect',abs(delta-effects[horizon]['effect'])<1e-12)
        check(horizon+' replication',math.ceil(factor*var/delta**2)==effects[horizon]['n80_per_cell'])
    path=OUT/'verification.json'
    if path.exists():raise RuntimeError('verification record already exists')
    write_json(path,dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,completed=date()))
    if not all(c['passed'] for c in checks):raise RuntimeError('artifact verification failed')
    print(f'PASS: {len(checks)} preservation and outcome checks')


if __name__=='__main__':
    functions={'validate':validate,'validate-correction1':lambda:validate('-correction1'),
               'pilot':run_pilot,'matched':run_matched,'complete-matched':complete_matched,'verify':verify}
    if len(sys.argv)!=2 or sys.argv[1] not in functions:
        raise SystemExit('usage: run_fix1.py validate|pilot|matched|verify')
    functions[sys.argv[1]]()
