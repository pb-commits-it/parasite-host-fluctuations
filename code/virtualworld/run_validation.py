"""Execute registered gates once, in order, stopping on the first failure.

Run: python3 -I virtualworld/run_validation.py
Independent references: ring algebra, supplied R16/memB archives, gamma
invariant density quadrature and a separately coded multitype branching PGF.
No model output is used to choose tolerances, parameters, seeds or run lengths.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0,str(HERE))
import ast,csv,json,time,subprocess,importlib.util,traceback,hashlib
import numpy as np
from world import Duration,Stage,Route,Model
from world.model import ring
from world import hosts
from world.invasion import products,threshold,ring_coefficient,resolvent_coefficient,gamma_exact,se
from world.simulation import simulate

RESULTS=HERE/'results'
BASE=(__import__('pathlib').Path(__file__).resolve().parent / '..').resolve()


def date():
    return subprocess.check_output(['date','+%Y-%m-%d %H:%M:%S %Z'],text=True).strip()


def record(rows,case,actual,expected,tolerance,**extra):
    error=abs(float(actual)-float(expected))
    rows.append(dict(case=case,actual=float(actual),expected=float(expected),
                     error=error,tolerance=float(tolerance),passed=bool(error<=tolerance),**extra))


def direction(rows,case,delta,standard_error=0.,floor=0.):
    limit=max(floor,4*standard_error)
    rows.append(dict(case=case,actual=float(delta),expected='positive',error=float(max(0,limit-delta)),
                     tolerance=float(limit),passed=bool(delta>limit),se=float(standard_error)))


def gate_i():
    rows=[]
    for idx,means in enumerate(([2,5],[1,2,7],[2,3,4,5])):
        m=ring(means); T=sum(means)
        # Independent one-step empirical mean of individual offspring counts.
        rng=np.random.default_rng(151100+idx)
        for src in range(m.n):
            n=np.zeros((50000,m.n),dtype=int); n[:,src]=1
            draws=m.step(n,np.ones((len(n),1)),rng)
            for dst in range(m.n):
                sd=draws[:,dst].std(ddof=1)/np.sqrt(len(n))
                record(rows,f'individual mean {means} {src}->{dst}',draws[:,dst].mean(),m.matrix()[dst,src],4*sd+.002,se=sd)
        for phi in (0.,.6):
            name=f'{means}, phi={phi}'
            K=ring_coefficient(means,phi)
            record(rows,'resolvent '+name,resolvent_coefficient(m,phi),K,1e-10)
            lower,upper=1/(2*T),(1+2*phi/(1-phi))/(2*T)
            record(rows,'band '+name,np.clip(K,lower,upper),K,1e-10)
            h=hosts.lognormal(65536+2048,32,.03,phi,151120+idx,antithetic=True)
            r,_,_=products(m,h,0.,2048,32,control=1/T,control_mean=1.)
            values=-T*r/.03**2
            record(rows,'linear limit '+name,values.mean(),K,.05*K+4*se(values),se=se(values),steps=65536,pairs=32)
            record(rows,'precision '+name,min(se(values),.03*K),se(values),0.)
            print('i',name,values.mean(),K,se(values),flush=True)
    return rows


def r16_model(row):
    E,S,A,P,M=ast.literal_eval(row['bits'])
    fine=row['fine']=='True'; dt=.25 if fine else 1.
    stay=float(row['stay'])/dt
    a=Stage('exposed',Duration.geometric(1/dt),.5*dt if E else 0.,mode='linear_delay' if E else 'waste')
    b=Stage('long',Duration.chain(stay,4) if S else Duration.geometric(stay),1-.99**dt if M else 0.,host=0)
    rr=[Route(0,1,host=i,share=1/(1+A)) for i in range(1+A)]+[Route(1,0,birth=True)]
    return Model([a,b],rr,1+A,4 if P else 1,1-.85**dt if P else 0.,1 if P else None)


def gate_ii():
    rows=[]
    path=BASE/'studies/engine-R16'
    spec=importlib.util.spec_from_file_location('reference_engine',path/'lifecycle_engine.py')
    ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    archive={r['id']:r for r in csv.DictReader((path/'R16_results.csv').open())}
    ids=['0c-00000','0c-01000','0c-00100','0c-00010','0c-00001','0f-10000']
    saved={}
    for idx,id_ in enumerate(ids):
        row=archive[id_];m=r16_model(row)
        E,S,A,P,M=ast.literal_eval(row['bits']);dt=.25 if row['fine']=='True' else 1.
        phi=float(row['phi'])**dt;sig=float(row['sigma'])
        ss=[ref.Stage('exposed',1/dt,death=.5*dt if E else 0.,contact=bool(E)),
            ref.Stage('long',float(row['stay'])/dt,'chain' if S else 'memoryless',4 if S else 1,1-.99**dt if M else 0.)]
        rr=[ref.Transfer('exposed','long',share=1/(1+A),driver=f'h{i}') for i in range(1+A)]+[ref.Transfer('long','exposed',birth=True)]
        ds=[ref.Driver(f'h{i}',sig,phi,'bounded',.05) for i in range(1+A)]
        desc=ref.Description(ss,rr,ds,patches=ref.Patches(4 if P else 1,0.,'long',1-.85**dt if P else 0.))
        eng=ref.Engine(desc).critical()
        test_h=hosts.bounded(10,3,sig,phi,151200+idx,m.drivers,m.patches)
        err=np.max(abs(m.matrix(test_h,m.critical_tilt())-eng.matrix(test_h)))
        record(rows,id_+' matrix',err,0.,1e-11)
        factor=int(1/dt); steps=32768*factor; burn=2048*factor
        h=hosts.bounded(steps+burn,32,sig,phi,151210+idx,m.drivers,m.patches)
        ans=threshold(m,h,burn)
        combined=np.hypot(ans['se'],float(row['se']))
        record(rows,id_+' D',ans['D'],float(row['D']),4*combined+.002,se=ans['se'],reference_se=float(row['se']),steps=steps)
        record(rows,id_+' precision',min(ans['se'],max(.003,.05*abs(ans['D']))),ans['se'],0.)
        record(rows,id_+' root',ans['residual'],0.,1e-7)
        record(rows,id_+' last half',ans['half_difference'],0.,4*ans['half_se']+.002,se=ans['half_se'])
        saved[id_]=ans
        print('ii',id_,ans['D'],row['D'],ans['se'],flush=True)
    (RESULTS/'validation_r16_roots.json').write_text(json.dumps(saved,indent=2))
    return rows


def gate_iii():
    rows=[];saved={}
    for idx,p in enumerate((2.,1.,.25)):
        h=hosts.gamma(65536+2048,64,p,151300+idx)
        prev=None
        for s in (.2,.5,.8,.95):
            label=f'p={p}, s={s}'
            exact=gamma_exact(s,p); refined=gamma_exact(s,p,True)
            record(rows,label+' quadrature',exact,refined,1e-8)
            tau=1/(1-s);m=ring([1,tau])
            ans=threshold(m,h,2048,control=1/(1+tau),control_mean=1.,initial_y=exact)
            record(rows,label+' D',ans['D'],exact,4*ans['se']+.0005,se=ans['se'])
            record(rows,label+' root',ans['residual'],0.,1e-7)
            if prev:
                direction(rows,label+' exact decrease',prev[0]-exact,floor=1e-8)
                dif=np.array(prev[1]['samples'])-ans['samples']
                direction(rows,label+' simulated decrease',dif.mean(),se(dif))
            prev=exact,ans;saved[label]=ans
            print('iii',label,ans['D'],exact,ans['se'],flush=True)
    (RESULTS/'validation_gamma_roots.json').write_text(json.dumps(saved,indent=2))
    return rows


def gate_iv():
    # The archive schema is checked explicitly, never inferred by row order.
    archive=list(csv.DictReader((BASE/'studies/memory-memB/memB_tables.csv').open()))
    rows=[];saved={}
    for ai,sigma in enumerate((.3,.6)):
        for phi in (.6,.9):
            h=hosts.lognormal(65536+2048,32,sigma,phi,151400+ai,antithetic=True)
            prev=None
            for tau in (2.,5.,10.):
                label=f'sigma={sigma}, phi={phi}, tau={tau}'
                q=1/tau
                coefficient=q*(1+2*phi/(1+q*phi))/(2*(1+q))
                record(rows,label+' memA',ring_coefficient([1,tau],phi),coefficient,1e-10)
                m=ring([1,tau])
                ans=threshold(m,h,2048,32,control=1/(1+tau),control_mean=1.,initial_y=coefficient*sigma*sigma)
                matches=[r for r in archive if r['model']=='lognormal' and abs(float(r['amplitude'])-sigma)<1e-8 and abs(float(r['phi_input'])-phi)<1e-8 and abs(float(r['tau'])-tau)<1e-8 and int(r['ring'])==2]
                if len(matches)!=1:
                    raise RuntimeError(f'expected one memB row for {label}, found {len(matches)}')
                ref=matches[0]
                record(rows,label+' D',ans['D'],float(ref['D']),4*np.hypot(ans['se'],float(ref['SE_D']))+.001,se=ans['se'],reference_se=float(ref['SE_D']))
                record(rows,label+' root',ans['residual'],0.,1e-7)
                if prev:
                    dif=np.array(prev['samples'])-ans['samples']
                    direction(rows,label+' decrease',dif.mean(),se(dif))
                prev=ans;saved[label]=ans
                print('iv',label,ans['D'],ref['D'],ans['se'],flush=True)
    (RESULTS/'validation_memory_roots.json').write_text(json.dumps(saved,indent=2))
    return rows


def branching_prediction(y,steps):
    """Independent PGF recursion for stays 2,5; each exit is Poisson.

No world event, matrix, or probability-generating function is called here.
q_i(t) is extinction by t from one individual of stage i. Initial ten are
all stage zero, so extinction by t is q_0(t)^10.
"""
    q=np.zeros(2);out={0:0.}
    for t in range(1,steps+1):
        a,b=q
        q=np.array([.5*a+.5*np.exp(b-1),.8*b+.2*np.exp(np.exp(y)*(a-1))])
        out[t]=float(q[0]**10)
    return out


def gate_v():
    rows=[]
    for idx,y in enumerate((-.1,0.,.1)):
        m=ring([2,5]);n=np.zeros((10000,2),int);n[:,0]=10
        out=simulate(m,n,120,151500+idx,hosts=np.ones((120,10000,1)),y=y)
        ref=branching_prediction(y,120)
        for t in (20,60,120):
            actual=np.mean(out['extinction_time']<=t);p=ref[t]
            sd=np.sqrt(p*(1-p)/10000)
            record(rows,f'logR0={y}, horizon={t}',actual,p,4*sd+.002,se=sd,replicates=10000)
            print('v',y,t,actual,p,flush=True)
        np.savez_compressed(RESULTS/f'validation_demography_{idx}.npz',extinction_time=out['extinction_time'],totals=out['totals'])
    return rows


def report(state):
    lines=['# Validation: vworld-1','',f"Run started {state['started']}; updated {date()}.",
           'All results below are numerical checks with the tolerances registered in protocol.md. The five gates run sequentially; failure prevents the pilot.','',
           'One-step individual-mean checks connect the stochastic sampler to the same expectation operator used for invasion. R16 is imported only as a matrix reference, never as the world implementation. Gamma quadrature and branching PGFs are separate algorithms.','']
    for key,value in state['gates'].items():
        lines += [f'## Gate {key}: {value["status"]}', '',f"{len(value['checks'])} comparisons; {value['seconds']:.2f} seconds.",'',
                  '| Check | Actual | Reference | Error | Tolerance | Result |','|---|---:|---:|---:|---:|---|']
        for r in value['checks']:
            expected=f"{r['expected']:.9g}" if isinstance(r['expected'],(int,float)) else r['expected']
            lines.append(f"| {r['case']} | {r['actual']:.9g} | {expected} | {r['error']:.3g} | {r['tolerance']:.3g} | {'pass' if r['passed'] else 'FAIL'} |")
        if value.get('exception'):lines += ['',value['exception']]
    lines += ['', 'For direction rows, actual is the preceding minus following D; tolerance is the required positive margin (four paired standard errors, or the quadrature floor). Error is any shortfall. Precision rows compare the observed SE to its registered ceiling and pass only below that ceiling.',
              '', 'Sampling errors are pointwise. Quadrature refinement, matrix agreement and half-run diagnostics are numerical controls, not interval-certified bounds on every systematic error.',
              '',f"Overall status: {state['status']}. Total measured compute: {state['seconds']:.2f} seconds."]
    (HERE/'validation.md').write_text('\n'.join(lines)+'\n')
    (RESULTS/'validation.json').write_text(json.dumps(state,indent=2))


def main():
    RESULTS.mkdir(exist_ok=True)
    if (RESULTS/'validation.json').exists():
        raise RuntimeError('validation is append-only: existing run must not be overwritten')
    start=time.monotonic()
    state=dict(started=date(),protocol_sha256=hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest(),gates={},status='running',seconds=0.)
    for key,fn in [('i',gate_i),('ii',gate_ii),('iii',gate_iii),('iv',gate_iv),('v',gate_v)]:
        t=time.monotonic();exception=None
        try:
            rows=fn()
        except Exception:
            rows=[];exception=traceback.format_exc()
        passed=bool(rows) and all(r['passed'] for r in rows) and exception is None
        if time.monotonic()-start>1800:
            passed=False;exception=(exception or '')+'\nValidation compute budget exceeded.'
        state['gates'][key]=dict(checks=rows,status='PASS' if passed else 'FAIL',seconds=time.monotonic()-t,exception=exception)
        state['seconds']=time.monotonic()-start
        state['status']='running' if passed else 'FAIL — pilot prohibited'
        report(state)
        print('GATE',key,state['gates'][key]['status'],flush=True)
        if not passed:
            (HERE/'pilot.md').write_text('# Pilot not run\n\nThe registered stop rule was triggered by validation gate '+key+'. See validation.md. No pilot results were computed.\n')
            return 1
    state['status']='PASS';report(state)
    return 0


if __name__=='__main__': sys.exit(main())
