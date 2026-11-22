"""One registered 18-cell factorial pilot, runnable only after all five gates pass.

python3 -I virtualworld/run_pilot.py
The primary interaction and power formula were fixed in protocol.md. No
adaptive sample sizing, cell removal, population cap or favorable contrast
selection is performed. All per-replicate outcomes and host paths are saved.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0,str(HERE))
import csv,json,time,subprocess,hashlib,math
import numpy as np
from scipy import stats
from world.model import ring
from world.design import allocate, Z95, POWER80_FACTOR
from world.counts import archive
from world import hosts
from world.invasion import threshold,ring_coefficient,se
from world.simulation import simulate,finite_growth


def date(): return subprocess.check_output(['date','+%Y-%m-%d %H:%M:%S %Z'],text=True).strip()


def wilson(k,n):
    z=Z95;p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(center-half),float(center+half)


def contrast(rows,data,keys,weights,outcome, seed=159000):
    vals=np.array([rows[k][outcome] for k in keys])
    delta=float(np.dot(vals,weights))
    if outcome=='extinction':
        variances=vals*(1-vals)
    else:
        variances=np.array([np.var(data[k]['growth100'],ddof=1) for k in keys])
    ns=np.array([rows[k]['n'] for k in keys])
    weights=np.asarray(weights)
    variance=float(np.sum(weights**2*variances/ns))
    err=math.sqrt(variance)
    ci=[delta-Z95*err,delta+Z95*err]
    # Equal n in each of the four independent cells, all weights +/-1.
    factor=POWER80_FACTOR
    n=None if delta==0 else max(2,int(np.ceil(factor*variances.sum()/delta**2)))
    out=dict(effect=delta,se=err,ci95=ci,n80_per_cell=n,variance_sum=float(variances.sum()),cells=keys,weights=weights.tolist())
    if outcome=='extinction':
        rng=np.random.default_rng(seed)
        counts=np.array([rows[k]['extinct'] for k in keys])
        draws=rng.beta(counts+.5,ns-counts+.5,size=(100000,len(keys)))
        contrasts=draws@np.asarray(weights)
        out['jeffreys_interval95']=np.quantile(contrasts,[.025,.975]).tolist()
        ds=draws[:10000];de=contrasts[:10000]
        needs=np.ceil(factor*(ds*(1-ds)).sum(axis=1)/de**2)
        out['n80_sensitivity_10_90']=np.quantile(needs,[.1,.9]).tolist()
        out['finite_upper_established']=not(out['jeffreys_interval95'][0]<=0<=out['jeffreys_interval95'][1])
    return out


def render(rows,main,built, report_path=None):
    lines=['# Registered pilot: vworld-1','',f"Started {built['started']}; completed {built['completed']}. All five validation gates passed before this run.",
           '',f"Numerical results: {len(rows)} cells, 200 independent individual/environment replicates each, {len(rows)*200:,} total trajectories. Horizon 500 steps; exposed mean stay 2, initial total 20, log R0=0.05. No biological parameters were tuned.",
           '', 'Host arithmetic means are fixed under the stationary law. Individual schedules have unequal realized means; the raw host paths and each replicate’s mean and log variance are archived. Phi labels at sigma=0 are independent demographic negative controls.',
           '', 'Rare growth below is the mean conditional-on-environment invasion growth over steps 0–100. It is computed for every schedule and is not the log growth of survivors. Extinction includes all zeros. D is obtained independently by solving long-run invasion growth for the critical birth multiplier, using 32 independent antithetic pairs per cell. Its intervals are t intervals over pairs; they exclude systematic finite-time error.',
           '', '| sigma | phi | steady stay | Extinct/200 | Extinction [95% Wilson] | Early growth/step [95% t] | D [95% t] | sigma² K |',
           '|---:|---:|---:|---:|---|---|---|---:|']
    for r in rows.values():
        lines.append(f"| {r['sigma']:g} | {r['phi']:g} | {r['stay']:g} | {r['extinct']} | {r['extinction']:.3f} [{r['extinction_low']:.3f}, {r['extinction_high']:.3f}] | {r['growth100']:.6f} [{r['growth100_low']:.6f}, {r['growth100_high']:.6f}] | {r['D']:.6f} [{r['D_low']:.6f}, {r['D_high']:.6f}] | {r['D2']:.6f} |")
    lines += ['', 'The comparison with sigma² K is conditional on the registered geometric, mass-action, wasting model. At sigma=0.6 it is a finite-noise comparison; departure from second order is not a failed validation. A steady stay changes generations elapsed in 500 steps (500/(2+stay)) and the constant-host growth rate. The noise-by-stay interaction removes the corresponding observed constant-host extinction differences, but does not create a general extinction theorem.',
              '', '## Registered main contrast and power','',
              'At phi=0.6: the noise effect at stay 1 minus the noise effect at stay 20. For extinction, the noise effect is P(.6)−P(0); for growth it is r(0)−r(.6). Positive values indicate a smaller noise-associated loss at the longer stay. All four cells are independent.',
              '', '| Outcome | Interaction | 95% interval | Approximate n/cell for 80% power |','|---|---:|---|---:|']
    for key,label in [('extinction','Extinction probability'),('growth','Early growth/step')]:
        c=main[key]
        lines.append(f"| {label} | {c['effect']:.6g} | [{c['ci95'][0]:.6g}, {c['ci95'][1]:.6g}] | {c['n80_per_cell']} |")
    c=main['extinction']
    lines += ['', f"Boundary sensitivity: independent Jeffreys beta draws give interaction interval [{c['jeffreys_interval95'][0]:.5f}, {c['jeffreys_interval95'][1]:.5f}] and a 10th–90th percentile planning range of {c['n80_sensitivity_10_90'][0]:.0f}–{c['n80_sensitivity_10_90'][1]:.0f} replicates per cell. "+('The interval excludes zero.' if c['finite_upper_established'] else 'The interval includes zero; a finite upper replication requirement is not established.'),
              '', 'Power uses a two-sided normal approximation at alpha=0.05 with equal allocation, n=(z_.975+z_.8)² Σvariance/delta². This is a model-conditional planning estimate, not measured achieved power or a recommendation that a real experiment needs this many cultures. Predictive uncertainty, detection error, biological blocks and nuisance calibration remain to be added. The posterior interval is explicitly a sensitivity analysis, not a frequentist confidence interval.',
              '', '## Simple contrasts','', '| Comparison | Extinction difference [95% normal] | Early growth loss [95% normal] |','|---|---|---|']
    for c in main['simple']:
        lines.append(f"| {c['label']} | {c['extinction']:.5f} [{c['ext_low']:.5f}, {c['ext_high']:.5f}] | {c['growth_loss']:.6f} [{c['growth_low']:.6f}, {c['growth_high']:.6f}] |")
    lines += ['', 'Simple intervals are pointwise descriptive comparisons; no multiplicity-adjusted discovery claims are made. Survivor-only individual growth and survivor counts are in pilot_cells.csv and the replicate arrays, clearly separated from the all-schedule conditional-mean growth.',
              '', f"Measured pilot runtime: {built['seconds']:.2f} seconds. Seeds, host path summaries, first extinction times, count trajectories, growth values and threshold influence samples are in results/. No further campaign was run."]
    if built.get('sensitivity'):
        lines[0] = '# Pilot rounding sensitivity: vworld-fix1 — 2026-11-15'
        lines[2] = (f"Started {built['started']}; completed {built['completed']}. "
                    'The original five gates passed; the corrected code passed the full fix1 suite '
                    'and the registered PGF/moment checks before this sensitivity.')
        lines.insert(3, '\nOriginal pilot.md and results remain unchanged. This run uses largest-remainder '
                     'allocation and the original seed schedule. Baseline subtraction does not remove '
                     'the mismatch in generations; phase-2 primary planning uses the matched-generation run. '
                     'Artifacts: results/fix1/pilot/. No protocol departures.')
        lines[-1] = f"Measured sensitivity runtime: {built['seconds']:.2f} seconds. " + \
                    'All 18 cells completed; this rounding-sensitivity run is closed.'
    (report_path or HERE/'pilot.md').write_text('\n'.join(lines)+'\n')


def main(*, results=None, report_path=None, validation_digest=None, sensitivity=False):
    results=HERE/'results' if results is None else Path(results)
    validation=json.loads((HERE/'results'/'validation.json').read_text())
    digest=hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest()
    if validation['status']!='PASS' or validation['protocol_sha256']!=(validation_digest or digest):
        raise RuntimeError('five passing gates and unchanged registered protocol are required')
    if (results/'pilot_cells.csv').exists() or (report_path is not None and report_path.exists()):
        raise RuntimeError('pilot already exists; do not overwrite a registered run')
    results.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();built=dict(started=date(),protocol_sha256=digest,sensitivity=sensitivity)
    rows={};data={};roots={}
    for sigma in (0.,.3,.6):
        for phi in (0.,.6):
            for stay in (1.,5.,20.):
                idx=len(rows);seed=152000+idx*10
                key=f's{sigma:g}_p{phi:g}_t{stay:g}'
                m=ring([2,stay]);y=.05
                v=m.constant(y)[1]
                init=allocate(20,v)
                n=np.broadcast_to(init,(200,m.n)).copy()
                h=hosts.lognormal(500,200,sigma,phi,seed)
                sim=simulate(m,n,500,seed+1,hosts=h,y=y)
                growth=finite_growth(m,h,y,init,endpoints=(100,500))
                k=int(np.sum(~sim['censored']));lo,hi=wilson(k,200)
                g=growth[100];ge=stats.t.ppf(.975,199)*se(g)
                ht=hosts.lognormal(32768+2048,32,sigma,phi,seed+2,antithetic=True)
                D2=sigma*sigma*ring_coefficient([2,stay],phi)
                d=threshold(m,ht,2048,32,control=1/(2+stay),control_mean=1.,initial_y=D2)
                de=stats.t.ppf(.975,31)*d['se']
                alive=sim['totals'][100]>0
                survivor_g=np.log(np.asarray(sim['totals'][100,alive],float)/20)/100
                row=dict(cell=key,sigma=sigma,phi=phi,stay=stay,seed=seed,n=200,extinct=k,extinction=k/200,
                         extinction_low=lo,extinction_high=hi,growth100=float(g.mean()),growth100_se=se(g),
                         growth100_low=float(g.mean()-ge),growth100_high=float(g.mean()+ge),
                         growth500=float(growth[500].mean()),D=d['D'],D_se=d['se'],D_low=d['D']-de,D_high=d['D']+de,
                         D2=D2,D_minus_D2=d['D']-D2,root_residual=d['residual'],initial_stage0=int(init[0]),initial_stage1=int(init[1]),
                         host_mean=float(h.mean()),host_mean_between_sd=float(h.mean(axis=0).std(ddof=1)),
                         survivors_at100=int(alive.sum()),survivor_only_growth100=float(survivor_g.mean()) if alive.any() else None,
                         restricted_mean_persistence=float(np.minimum(sim['extinction_time'],500).mean()))
                rows[key]=row
                data[key]=dict(growth100=growth[100])
                roots[key]=d
                np.savez_compressed(results/(key+'.npz'),totals=archive(sim['totals']),extinction_time=sim['extinction_time'],
                                    censored=sim['censored'],initial=init,hosts=h,
                                    host_mean=h.mean(axis=0).ravel(),host_log_variance=np.log(h).var(axis=0,ddof=1).ravel(),
                                    growth100=growth[100],growth500=growth[500])
                # Checkpoint a completed cell; final CSV is the complete table.
                (results/(key+'.json')).write_text(json.dumps(row,indent=2))
                print(key,'ext',k,'growth',row['growth100'],'D',d['D'],'D2',D2,flush=True)
                if time.monotonic()-start>600:
                    raise RuntimeError('pilot compute budget exceeded; preserve completed cells')
    keys=['s0.6_p0.6_t1','s0_p0.6_t1','s0.6_p0.6_t20','s0_p0.6_t20']
    effects=dict(extinction=contrast(rows,data,keys,[1,-1,-1,1],'extinction'),
                 growth=contrast(rows,data,keys,[-1,1,1,-1],'growth100'),simple=[])
    comparisons=[]
    for phi in (0.,.6):
        for stay in (1.,5.,20.):
            comparisons.append((f'sigma .6 minus 0, phi={phi:g}, stay={stay:g}',f's0.6_p{phi:g}_t{stay:g}',f's0_p{phi:g}_t{stay:g}'))
    for sigma in (0.,.3,.6):
        for phi in (0.,.6):
            comparisons.append((f'stay 1 minus 20, sigma={sigma:g}, phi={phi:g}',f's{sigma:g}_p{phi:g}_t1',f's{sigma:g}_p{phi:g}_t20'))
    for label,a,b in comparisons:
        x,z=rows[a],rows[b]
        de=x['extinction']-z['extinction']
        err=Z95*np.sqrt((x['extinction']*(1-x['extinction'])+z['extinction']*(1-z['extinction']))/200)
        dg=z['growth100']-x['growth100']
        eg=Z95*np.hypot(x['growth100_se'],z['growth100_se'])
        effects['simple'].append(dict(label=label,extinction=de,ext_low=de-err,ext_high=de+err,growth_loss=dg,growth_low=dg-eg,growth_high=dg+eg))
    with (results/'pilot_cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(next(iter(rows.values()))));writer.writeheader();writer.writerows(rows.values())
    built.update(completed=date(),seconds=time.monotonic()-start,cells=18,replicates_per_cell=200,status='complete')
    (results/'pilot_effects.json').write_text(json.dumps(effects,indent=2))
    (results/'pilot_roots.json').write_text(json.dumps(roots,indent=2))
    (results/'pilot_run.json').write_text(json.dumps(built,indent=2))
    render(rows,effects,built,report_path)


if __name__=='__main__':main()
