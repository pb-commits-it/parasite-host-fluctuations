"""Run the prewritten protocol. python3 -I run_R16.py [--pilot | --report-only].

Three processes; checkpoint each completed cell. Every report value is printed
by this script. Invalid requested combinations remain explicit table rows.
"""
from pathlib import Path
import os
# Set before numpy is imported, including spawned worker interpreters.
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import sys, importlib.util, time, json, csv, math, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from scipy.stats import spearmanr
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec=importlib.util.spec_from_file_location('lifecycle_engine',HERE/'lifecycle_engine.py')
e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
Stage,Transfer,Driver,Patches,Description,Engine=e.Stage,e.Transfer,e.Driver,e.Patches,e.Description,e.Engine
SWITCHES='ESAPM'
CYCLES=[('long adult stage',24.,.6,.4),('long adult stage',24.,1.4,.4),('moderate',7.5,1.4,.75),('short',2.3,.7,.3),('short',2.3,1.2,.3)]
SETTINGS={'pairs':8,'steps_coarse':12000,'V_environments':16,'V_lines':96,'extinction_trials':128}


def model_description(job,death_override=None):
    E,S,A,P,M=job['bits'];fine=job['fine'];dt=1/4 if fine else 1.
    wait=1/dt;long=job['stay']/dt
    death=(1-.99**dt) if (M if death_override is None else death_override) else 0.
    stage0=Stage('exposed',wait,'memoryless' if fine or E else 'fixed',death=.5/wait if E else 0.,contact=bool(E))
    stage1=Stage('long',long,'chain' if S else 'memoryless',4 if S else 1,death)
    phi=job['phi']**dt
    drivers=[Driver('h'+str(k),job['sigma'],phi,'bounded',.05) for k in range(1+A)]
    transfers=[Transfer('exposed','long',1/(1+A),driver=d.name) for d in drivers]
    transfers.append(Transfer('long','exposed',birth=True))
    patches=Patches(4 if P else 1,0.,'long',1-.85**dt if P else 0.,'island',0.)
    return Description([stage0,stage1],transfers,drivers,patches=patches)


def jobs():
    out=[]
    for cycle,(name,stay,sigma,phi) in enumerate(CYCLES):
        for fine in (False,True):
            for bits in itertools.product((0,1),repeat=5):
                out.append({'id':f'{cycle}{"f" if fine else "c"}-'+''.join(map(str,bits)),
                            'cycle':cycle,'name':name,'stay':stay,'sigma':sigma,'phi':phi,
                            'fine':fine,'bits':list(bits),'seed':816000+len(out)*11})
    return out


def run_cell(job,settings):
    start=time.monotonic();row=dict(job)
    try:
        raw=Engine(model_description(job));model=raw.critical()
    except e.ModelError as error:
        return {**row,'status':'inadmissible','reason':str(error),'wall_seconds':time.monotonic()-start}
    off=Engine(model_description(job,death_override=False)).critical()
    unraised=Engine(model_description(job),birth_scale=off.birth_scale)
    row.update(status='ok',period=model.period,mean_lambda=raw.growth_factor,
               raw_log_R0=math.log(raw.R0),critical_birth_scale=model.birth_scale,
               death_cost=-math.log(unraised.R0),generation_steps=1/model.b,
               J2=model.jensen(quadratic=True)/model.b)
    try:row['D2']=model.coefficient()['D2']
    except e.PeriodicModelError:row['D2']=None
    factor=4 if job['fine'] else 1
    steps=settings['steps_coarse']*factor
    burn=max(1000*factor,int(math.ceil(10/model.b)))
    result=model.loss(steps=steps,burn=burn,pairs=settings['pairs'],seed=job['seed'])
    result.pop('samples')
    row.update(result)
    row['bound_pass']=row['D']<=row['J']+4*row['se']+1e-7
    if not row['bound_pass']:
        raise RuntimeError(f'BOUND FAILURE {row}')
    v=model.lineage_variance(steps=max(500*factor,int(math.ceil(10/model.b))),
                            environments=settings['V_environments'],lines=settings['V_lines'],seed=job['seed']+1)
    row.update(V=v['V_generation'],V_se=v['se_generation'],V_steps=v['steps'],
               V_environments=v['environments'],V_lines=v['lines'])
    if job['name']=='moderate':
        initial=np.zeros(model.n,np.int64)
        for p in range(model.description.patches.count):initial[p*model.k]=10000//model.description.patches.count
        ext=model.extinction(initial,int(math.ceil(2000/model.b)),settings['extinction_trials'],job['seed']+2,y=row['D']+.02)
        row.update(extinction=ext['fraction'],extinction_se=ext['se'],extinction_conditional=ext['conditional_mean'],
                   extinction_trials=ext['trials'],extinction_steps=ext['steps'],initial=ext['initial'])
    row['wall_seconds']=time.monotonic()-start
    return row


def fnum(x):
    return '—' if x is None else f'{x:.6g}'


def contrast(rows,index,interaction=None):
    lookup={tuple(r['bits']):r for r in rows if r['status']=='ok'}
    remaining=[i for i in range(5) if i not in [index,interaction]]
    values=[];variances=[]
    for bg in itertools.product((0,1),repeat=len(remaining)):
        fixed=dict(zip(remaining,bg));needed=[]
        combinations=[(0,),(1,)] if interaction is None else [(0,0),(0,1),(1,0),(1,1)]
        weights=[-1.,1.] if interaction is None else [1.,-1.,-1.,1.]
        for combo in combinations:
            bits=[fixed.get(i,0) for i in range(5)];bits[index]=combo[0]
            if interaction is not None:bits[interaction]=combo[1]
            needed.append(lookup.get(tuple(bits)))
        if any(r is None for r in needed):continue
        values.append(sum(w*r['D'] for w,r in zip(weights,needed)))
        variances.append(sum(w*w*r['se']**2 for w,r in zip(weights,needed)))
    if not values:return None
    return {'effect':float(np.mean(values)),'se':math.sqrt(sum(variances))/len(values),'blocks':len(values),
            'full':len(values)==2**len(remaining)}


def comparison(a,b):
    if a is None or b is None or a['status']!='ok' or b['status']!='ok':return ('unavailable',None,None)
    change=a['D']-b['D'];se=math.hypot(a['se'],b['se'])
    status='raises loss' if change>4*se else ('lowers loss' if change < -4*se else 'unresolved')
    return status,change,se


def build_report(rows,metadata):
    rows=sorted(rows,key=lambda r:(r['cycle'],r['fine'],r['bits']))
    good=[r for r in rows if r['status']=='ok'];invalid=[r for r in rows if r['status']!='ok']
    negative=[r for r in good if r['D']<0];gains=[r for r in negative if r['D']+4*r['se']<0]
    wrong=[r for r in good if r['D2'] is not None and r['D2']*r['D']<0]
    wrong_res=[r for r in wrong if abs(r['D'])>4*r['se']]
    poor=[r for r in good if r['D2'] is not None and abs(r['D2']-r['D'])>.3*abs(r['D'])]
    poor_res=[r for r in poor if abs(r['D2']-r['D'])>.3*abs(r['D'])+4*r['se']]
    bounds=[r for r in good if not r['bound_pass']]
    transfer_bad=[r for r in good if r['D']>r['J_transfer']+4*r['se']+1e-7]
    lookup={(r['cycle'],r['fine'],tuple(r['bits'])):r for r in rows}
    effects=[];interactions=[]
    for cycle in range(len(CYCLES)):
        for fine in (False,True):
            subset=[r for r in rows if r['cycle']==cycle and r['fine']==fine]
            for i in range(5):
                c=contrast(subset,i)
                effects.append({'cycle':cycle,'fine':fine,'switch':SWITCHES[i],**(c or {'effect':None,'se':None,'blocks':0,'full':False})})
            for i,j in itertools.combinations(range(5),2):
                c=contrast(subset,i,j)
                if c and abs(c['effect'])>4*c['se']:
                    interactions.append({'cycle':cycle,'fine':fine,'switch':SWITCHES[i]+SWITCHES[j],**c})
    parking=[];routes=[];shape=[]
    for fine in (False,True):
        for bits in itertools.product((0,1),repeat=5):
            a=lookup[(1,fine,bits)];b=lookup[(2,fine,bits)]
            status,delta,se=comparison(a,b)
            parking.append({'grid':'fine' if fine else 'coarse','bits':''.join(map(str,bits)),
                            'status':status,'change':delta,'se':se})
    for cycle in range(len(CYCLES)):
        for fine in (False,True):
            for index,destination in ((1,shape),(2,routes)):
                for bg in itertools.product((0,1),repeat=4):
                    bits=list(bg);bits.insert(index,0);b=lookup[(cycle,fine,tuple(bits))]
                    bits[index]=1;a=lookup[(cycle,fine,tuple(bits))]
                    status,delta,se=comparison(a,b)
                    destination.append({'cycle':cycle,'grid':'fine' if fine else 'coarse','on_bits':''.join(map(str,bits)),
                                        'status':status,'change':delta,'se':se})
    ext=[r for r in good if 'extinction' in r]
    reversals=[];concordant=0;compared=0
    for a,b in itertools.combinations(ext,2):
        if a['fine']!=b['fine']:continue
        dd=a['D']-b['D'];ds=math.hypot(a['se'],b['se'])
        de=a['extinction']-b['extinction'];es=math.hypot(a['extinction_se'],b['extinction_se'])
        if abs(dd)>4*ds and abs(de)>4*es:
            compared+=1
            if dd*de<0:reversals.append((a['id'],b['id'],dd,de))
            else:concordant+=1
    def smallest(items):
        if not items:return []
        degree=min(sum(r['bits']) for r in items)
        return [r['id'] for r in items if sum(r['bits'])==degree]
    def count_status(items,status):return sum(r['status']==status for r in items)
    def smallest_comparisons(items,status):
        selected=[r for r in items if r['status']==status]
        if not selected:return 'none'
        degree=min((r.get('on_bits') or r['bits']).count('1') for r in selected)
        return ', '.join(f"{r.get('cycle','long/moderate')}/{r['grid']}/{r.get('on_bits') or r['bits']}" for r in selected if (r.get('on_bits') or r['bits']).count('1')==degree)
    summary=[
        f"Illustrative requested cells: {len(rows)}; computed: {len(good)}; inadmissible: {len(invalid)}.",
        f"Negative D estimates: {len(negative)}; gains resolved at four SE: {len(gains)}; smallest resolved gain cells: {', '.join(smallest(gains)) or 'none'}.",
        f"Wrong second-order signs: {len(wrong)}; resolved wrong signs: {len(wrong_res)}.",
        f"Second-order size errors above 30 percent: {len(poor)}; resolved with the extra four-SE allowance: {len(poor_res)}.",
        f"At matched sigma, long versus moderate raises loss in {count_status(parking,'raises loss')} backgrounds, lowers it in {count_status(parking,'lowers loss')}, and is unavailable in {count_status(parking,'unavailable')}; memory differs.",
        f"The four-part chain raises loss in {count_status(shape,'raises loss')} available comparisons and lowers it in {count_status(shape,'lowers loss')}.",
        f"Alternative hosts lower loss in {count_status(routes,'lowers loss')} comparisons and raise it in {count_status(routes,'raises loss')}.",
        f"Resolved two-switch interactions: {len(interactions)}; these are conditional when cells are missing.",
        f"Edge-bound failures: {len(bounds)}; transfer-only exceedances: {len(transfer_bad)}.",
        f"Resolved extinction-order reversals: {len(reversals)} of {compared} resolved comparisons; measured at a matched excess log reproduction of 0.02."
    ]
    lines=['# R16: mechanisms together','', 'All life cycles and parameters are illustrative. No empirical inference is made.','']
    lines+=summary+['',f"The run used {metadata['workers']} processes and took {metadata['wall_seconds']:.3f} seconds. Settings: {json.dumps(metadata['settings'],sort_keys=True)}.",
                    'The full factorial was requested but is not mathematically admissible everywhere. Missing cells prevent full five-switch effects for some cycles and grids. We report balanced contrasts on the available backgrounds. The protocol was written before running.',
                    'The coarse exposed stage is fixed for one step. Fine steps use a memoryless exposed stage. A four-part chain has geometric sub-stages; it is not a fixed total delay. None of the admissible requested cells is periodic. Fixed-timing periodic cycles were tested separately.',
                    'For the bounded host, sigma labels a matched lognormal log standard deviation. Its variance is exp(sigma squared) minus one. Its lower level is 0.05. Higher moments and raw-host autocovariances differ from a lognormal AR host.',
                    'The positive-stay contact constraint is checked at every possible bounded host level. With a fine mean wait of 4 and half dying, contact and death probabilities are each 0.125. Thus the host upper level must be below 7. Any nonnegative mean-one host then has variance below 6. At sigma 1.4 the requested variance exceeds this limit. This impossibility holds beyond the particular chosen two-state law.',
                    'A fixed one-step exposure does not alone make a cycle periodic. A memoryless long stage can supply random timing. The coarse short four-part chain is impossible because its mean would be less than its number of sub-stages.',
                    '', '## Meaning of the quantities','',
                    'D is the log birth multiplier restoring stochastic zero growth after calibration of the arithmetic-mean matrix. D2 is the resolvent expansion in sigma. J is the finite-amplitude edge Jensen bound, including contact staying edges. J2 is its quadratic part. V is measured from first-order rewards on reference paths, per reference birth. Therefore D2=J2-V/2 is the identity being checked; substituting finite-amplitude J is incorrect.',
                    'Standard errors describe sampling error across independent antithetic pairs. They do not bound finite-horizon bias. A negative point estimate is distinguished from a resolved gain. Numerical roots are not exact finite-noise boundaries.',
                    'Death cost is the loss in mean log R0 when births are left at their no-death values. Adding this cost to D gives the total required birth increase against that baseline. The table also reports the variability loss after births compensate the mean death cost.',
                    'Extinction is sampled as an exact conditional branching-process endpoint. The backward Poisson generating function produces the extinction probability in each environment. An independent Bernoulli draw produces the reported fraction and binomial standard error. This avoids overflow without capping populations. It does not record full forward trajectories.',
                    'The endpoint justification is by conditioning on one environmental history. A route with Poisson yield mu has generating function exp(mu times (q_move minus one)); continuation has generating function q_move. The weighted sum over mutually exclusive parental outcomes gives the one-step function. Induction gives its backward composition. Independent starting individuals then give the product of the resulting stage probabilities, each raised to its initial count. Sampling a Bernoulli variable with this probability has the branching process extinction law.',
                    '', '## Full table','',
                    'Cell names start with the cycle index, then c for coarse or f for fine, followed by ESAPM bits. Cycle key:',
                    '| Index | Illustrative cycle | Long stay | Sigma | Memory |','|---|---|---:|---:|---:|']
    for i,(name,stay,sig,phi) in enumerate(CYCLES):lines.append(f'| {i} | {name} | {stay:g} | {sig:g} | {phi:g} |')
    lines+=['','| Cell | J | J2 | D2 | D | SE(D) | V | SE(V) | Death cost | Extinct | Binomial SE | Status |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['status']=='ok':
            keys=['J','J2','D2','D','se','V','V_se','death_cost','extinction','extinction_se']
            lines.append('| '+r['id']+' | '+' | '.join(fnum(r.get(k)) for k in keys)+' | computed |')
        else:lines.append('| '+r['id']+' | '+' | '.join(['—']*10)+' | '+r['reason']+' |')
    lines+=['','Unrounded fields, raw mean growth, raw log R0, birth scales, root residuals, seeds, run lengths and extinction horizons are in `R16_results.csv` and `R16_results.json`. Every row there is also emitted in the text output.','',
            '## Main effects on D','',
            'Each contrast averages matched on-minus-off differences. A full main effect has all 16 backgrounds. Fewer backgrounds mean a conditional effect. An absent contrast is unavailable, not zero.',
            '| Cycle | Grid | Switch | Effect | SE | Matched backgrounds | Full |','|---|---|---|---:|---:|---:|---|']
    for r in effects:lines.append(f"| {r['cycle']} | {'fine' if r['fine'] else 'coarse'} | {r['switch']} | {fnum(r['effect'])} | {fnum(r['se'])} | {r['blocks']} | {r['full']} |")
    lines+=['','## Two-way interactions exceeding four standard errors','',
            'The contrast is 11 minus 10 minus 01 plus 00. A full contrast averages all 8 backgrounds.',
            '| Cycle | Grid | Switches | Interaction | SE | Backgrounds | Full |','|---|---|---|---:|---:|---:|---|']
    for r in interactions:lines.append(f"| {r['cycle']} | {'fine' if r['fine'] else 'coarse'} | {r['switch']} | {r['effect']:.6g} | {r['se']:.6g} | {r['blocks']} | {r['full']} |")
    lines+=['','## Negative losses and expansion failures','',
            '| Cell | D | SE | D2 | Negative estimate | Resolved gain | Wrong sign | Size error above 30 percent | Resolved size error |',
            '|---|---:|---:|---:|---|---|---|---|---|']
    for r in good:
        if r not in negative and r not in wrong and r not in poor:continue
        lines.append(f"| {r['id']} | {r['D']:.6g} | {r['se']:.6g} | {fnum(r['D2'])} | {r in negative} | {r in gains} | {r in wrong} | {r in poor} | {r in poor_res} |")
    lines+=['','## Parking in every background','',
            'Long minus moderate at matched sigma 1.4. Their host memories differ. These are descriptive cycle comparisons, not isolated tests of duration. Every ESAPM background is included.',
            '| Grid | ESAPM | Change | SE | Result |','|---|---|---:|---:|---|']
    for r in parking:lines.append(f"| {r['grid']} | {r['bits']} | {fnum(r['change'])} | {fnum(r['se'])} | {r['status']} |")
    for title,items in [('Stay shape',shape),('Alternative hosts',routes)]:
        lines+=['',f'## {title} in every available comparison','',
                'The on cell is compared with the identical background with this switch off. Unavailable pairs are retained.',
                '| Cycle | Grid | On ESAPM | Change | SE | Result |','|---|---|---|---:|---:|---|']
        for r in items:lines.append(f"| {r['cycle']} | {r['grid']} | {r['on_bits']} | {fnum(r['change'])} | {fnum(r['se'])} | {r['status']} |")
    lines+=['','## Extinction ordering','',
            f'There are {len(ext)} moderate-cycle estimates. There are {compared} pair comparisons resolved in both quantities: {concordant} agree in order and {len(reversals)} reverse. Pairs are compared within the same time grid. All means are set to D+0.02, so this is an ordering at equal excess over the estimated stochastic threshold.',
            f"In {sum(r['se']>=.02 for r in ext)} extinction cells, one standard error of the fitted D is at least the intended 0.02 margin. In {sum(4*r['se']>=.02 for r in ext)} cells, four standard errors reach that margin. Birth-tilt uncertainty can therefore affect the ordering.",
            'The binomial errors condition on the fitted birth tilt. They do not include uncertainty in D. A comparison with extinction fractions at zero or one can have a zero plug-in error; such endpoint errors do not prove certainty.']
    for fine in (False,True):
        subset=[r for r in ext if r['fine']==fine]
        rho=float(spearmanr([r['D'] for r in subset],[r['extinction'] for r in subset]).statistic)
        lines.append(f"The {'fine' if fine else 'coarse'} rank correlation is {rho:.6g}.")
    lines+=['','| Cell A | Cell B | D(A)-D(B) | Extinction(A)-Extinction(B) |','|---|---|---:|---:|']
    for a,b,dd,de in reversals:lines.append(f'| {a} | {b} | {dd:.6g} | {de:.6g} |')
    lines+=['','## Four conclusions','',
            ('Cost fails in this admissible grid: variability can help. The smallest resolved gain cells are '+', '.join(smallest(gains))+'.') if gains else 'Cost is not refuted by a resolved negative loss in this grid.',
            'Parking is conditional. The shape comparisons show whether making a stay more regular removes protection. The matched-sigma cycle comparison also changes memory, so it cannot establish an exact duration boundary. Direct death costs remain even when the variability loss is small.',
            'The smallest available shape comparisons that raise loss are '+smallest_comparisons(shape,'raises loss')+'. The smallest descriptive long-versus-moderate reversals are '+smallest_comparisons(parking,'raises loss')+'.',
            'The number of sequential hosts is not varied by R16. The engine checks its second-order additivity separately. Alternative hosts are varied here; their effect is reported in every background. An alternative-route benefit is consistent with the qualified conclusion in the notes, not a failure of it.',
            'The smallest alternative-route protections are '+smallest_comparisons(routes,'lowers loss')+'. The smallest adverse alternative-route comparisons are '+smallest_comparisons(routes,'raises loss')+'.',
            'The edge bound survives all computed cells.' if not bounds else 'The edge bound has a numerical failure and interpretation must stop.',
            ('The transfer-only bound is exceeded in '+', '.join(r['id'] for r in transfer_bad)+'. Contact changes staying, so these exceedances fall outside that bound\'s assumptions.') if transfer_bad else 'No resolved transfer-only exceedance occurs in this grid. This does not extend that bound to contact limitation.',
            'No smallest breaking combination for the bound is found. No smallest breaking combination for the sequential-host equivalence is tested. Smallest grid examples are not globally minimal mechanisms.',
            '', '## Not done','',
            f'{len(invalid)} requested cells could not be run because their transition probabilities or stage durations are inadmissible. No replacement time step or variance convention was silently introduced.',
            'The fully crossed five-switch question is consequently not answered at every requested parameter set. Exact continuous boundaries, uniqueness of any boundary, and a general proof about combinations are not supplied.',
            'There is no controlled duration-only comparison across all three illustrative cycles at the same host variance and memory. The supplied cycles share only one sigma pair, and that pair has different memory.',
            'The run does not use empirical parameters, nonlinear host feedback, continuous time, or density dependence. It does not estimate a rigorous finite-horizon error bound. V has Monte Carlo error and represents the quadratic reward, not all finite-amplitude cumulants.',
            'Extinction is evaluated only for admissible moderate cells. It uses the requested Poisson law. Negative-binomial and shared-group laws have mean-matrix checks, but their long-horizon extinction is not part of R16.',
            'No finite-noise formula is claimed proved. The exact discrete admissibility obstruction follows from h squared <= upper_bound times h, hence Var(h)<=upper_bound-1 for mean-one nonnegative h. The strict staying margin makes the contact upper bound strict.',
            '']
    report='\n'.join(lines)
    (HERE/'report_R16.md').write_text(report)
    (HERE/'R16_summary.json').write_text(json.dumps({'summary':summary,'effects':effects,'interactions':interactions,'parking':parking,'shape':shape,'routes':routes,'gains':[r['id'] for r in gains],'wrong_sign':[r['id'] for r in wrong],'poor':[r['id'] for r in poor],'extinction_reversals':reversals},indent=2))
    columns=sorted(set().union(*(r.keys() for r in rows)))
    with (HERE/'R16_results.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    print('\nSCRIPT-GENERATED REPORT\n'+report,flush=True)
    return summary


class Tee:
    def __init__(self,file):self.file=file
    def write(self,s):sys.__stdout__.write(s);self.file.write(s);self.file.flush()
    def flush(self):self.file.flush()


def main():
    if not (HERE/'protocol_R16.md').exists():raise RuntimeError('write protocol before running')
    if '--report-only' in sys.argv:
        data=json.loads((HERE/'R16_results.json').read_text())
        output=HERE/'run_R16_output.txt'
        if output.exists():
            original=HERE/'run_R16_output_initial.txt'
            if not original.exists():output.rename(original)
        with output.open('w') as f:
            sys.stdout=Tee(f)
            print('Report regenerated from stored R16 results; simulations were not rerun.')
            print('Recorded run metadata:',json.dumps(data['metadata'],sort_keys=True))
            for row in data['rows']:print(json.dumps(row,sort_keys=True))
            build_report(data['rows'],data['metadata'])
            print('R16 COMPLETE recorded wall seconds',data['metadata']['wall_seconds'])
            sys.stdout=sys.__stdout__
        return
    if '--pilot' in sys.argv:
        selected=next(j for j in jobs() if j['cycle']==2 and j['fine'] and j['bits']==[0,1,1,1,1])
        row=run_cell(selected,SETTINGS)
        print(json.dumps(row,indent=2));return
    checks=json.loads((HERE/'test_engine_results.json').read_text())
    if not all(r['pass'] for r in checks['checks']):raise RuntimeError('tests must pass before R16')
    target=HERE/'run_R16_output.txt'
    if target.exists():
        i=1
        while (HERE/f'run_R16_output_attempt{i}.txt').exists():i+=1
        target.rename(HERE/f'run_R16_output_attempt{i}.txt')
    start=time.monotonic();rows=[]
    with target.open('w') as f:
        sys.stdout=Tee(f)
        print('R16 protocol fixed before running. Three processes. Settings:',SETTINGS,flush=True)
        context=mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=3,mp_context=context) as pool:
            futures={pool.submit(run_cell,j,SETTINGS):j for j in jobs()}
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                print(json.dumps(row,sort_keys=True),flush=True)
                metadata={'workers':3,'settings':SETTINGS,'wall_seconds':time.monotonic()-start}
                (HERE/'R16_checkpoint.json').write_text(json.dumps({'rows':rows,'metadata':metadata}))
                if metadata['wall_seconds']>2400:raise RuntimeError('40-minute runtime limit exceeded')
        metadata={'workers':3,'settings':SETTINGS,'wall_seconds':time.monotonic()-start}
        (HERE/'R16_results.json').write_text(json.dumps({'rows':rows,'metadata':metadata},indent=2))
        build_report(rows,metadata)
        print('R16 COMPLETE wall seconds',time.monotonic()-start,flush=True)
        sys.stdout=sys.__stdout__

if __name__=='__main__':main()
