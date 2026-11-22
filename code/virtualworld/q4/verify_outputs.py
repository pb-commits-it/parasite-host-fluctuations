"""Audit saved outcomes with independent ring matrices and frequency response.

No new biological experiment is run. Known algebra is checked before this
independent representation is used to audit real-record predictions.
"""
import sys,collections
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();sys.path.insert(0,str(P))
from numerics import *

def independent_parts(k,s):
    tau=np.array(means(k,s),float);T=tau.sum();v=tau/T
    A=np.diag(1-1/tau)
    for i in range(k):A[(i+1)%k,i]+=1/tau[i]
    Q=np.eye(k)-v[:,None];B=np.zeros((k,k));B[1,0]=1/tau[0]
    return A,Q,B,v,T

def response(k,s,p):
    A,Q,B,v,T=independent_parts(k,s)
    return 1/(2*T)-T*p*(np.ones(k)@B@np.linalg.solve(np.eye(k)-p*A@Q,Q@B@v))

def direct_full_prediction(k,s,h):
    z=np.log(h);z-=z.mean();spectrum=abs(np.fft.fft(z))**2/len(z)**2
    return float(sum(power*response(k,s,np.exp(2j*np.pi*j/len(z))).real for j,power in enumerate(spectrum)))

def monodromy_error(k,s,h,D):
    tau=np.array(means(k,s),float);product=np.eye(k)
    for val in h:
        A=np.diag(1-1/tau)
        for i in range(k):A[(i+1)%k,i]+=(val if i==0 else (np.exp(D) if i==k-1 else 1.))/tau[i]
        product=A@product
    return float(abs(np.log(max(np.linalg.eigvals(product).real))))

if __name__=='__main__':
    dest=P/'verification.json'
    if dest.exists():raise RuntimeError('append-only verification')
    start=time.monotonic();checks=[]
    def check(name,ok,**detail):checks.append(dict(check=name,passed=bool(ok),**detail))
    # Reproduce known AR ring coefficients with an independent matrix setup first.
    for k in (2,3):
        for s in (1,5,20):
            for phi in (-.5,0,.6,.9):
                e=abs(response(k,s,phi)-ring_coefficient(means(k,s),phi))
                check(f'known AR response {k}/{s}/{phi}',e<=1e-10,error=float(e),tolerance=1e-10)
    if not all(r['passed'] for r in checks):raise RuntimeError('independent audit reference failed')
    records=json.loads((P/'raw/selected-trajectories.json').read_text());raw={r['series_id']:r for r in records}
    cells=[json.loads(s) for s in (P/'cells.jsonl').read_text().splitlines()];ci={(r['series_id'],r['stages'],r['stay']):r for r in cells}
    check('1146 unique cells',len(ci)==1146 and len(cells)==1146)
    maxroot=maxfreq=0.
    for r in cells:
        h=norm_host(raw[r['series_id']]['values']);k=r['stages'];s=r['stay'];e=monodromy_error(k,s,h,r['D']);maxroot=max(maxroot,e)
        check('direct unnormalized root '+r['series_id']+f'/{k}/{s}',e<=1e-7,error=e,tolerance=1e-7)
        if np.all(h>0):
            e=abs(direct_full_prediction(k,s,h)-r['prediction_full']);maxfreq=max(maxfreq,e)
            check('independent Fourier prediction '+r['series_id']+f'/{k}/{s}',e<=1e-10,error=e,tolerance=1e-10)
        else:check('zero prediction unavailable '+r['series_id']+f'/{k}/{s}',r['prediction_full'] is None and r['prediction_AR'] is None)
        for name in ('AR','full'):check('H1 status '+r['series_id']+f'/{k}/{s}/{name}',r['H1_'+name]==status(r['D'],r['prediction_'+name]))
    persistence=list(csv.DictReader((P/'persistence-exact.csv').open()));pi={(r['series_id'],int(r['stages']),int(r['stay']),r['endpoint']):r for r in persistence}
    check('4584 persistence endpoints',len(pi)==4584)
    for sid in raw:
        with np.load(P/'phase-persistence'/f'{sid}.npz') as phase:
            for k in (2,3):
                for s in (1,5,20):
                    T=sum(means(k,s));last=np.ones(len(raw[sid]['values']))
                    for label,t in sorted({'G23':23*T,'G50':50*T,'G100':100*T,'step500':500}.items(),key=lambda z:z[1]):
                        row=pi[sid,k,s,label];p=phase[f'k{k}_s{s}_{label}']
                        check('PGF saved '+sid+f'/{k}/{s}/{label}',int(row['horizon'])==t and abs(float(row['survival'])-p.mean())<1e-14 and
                            abs(float(row['phase_min'])-p.min())<1e-14 and abs(float(row['phase_max'])-p.max())<1e-14 and np.all(p<=last+1e-12) and np.all(p>=-1e-12))
                        last=p
    mc=list(csv.DictReader((P/'persistence.csv').open()));mci={(r['series_id'],int(r['stages']),int(r['stay']),r['endpoint']):r for r in mc}
    for path in sorted((P/'demography').glob('*.npz')):
        with np.load(path) as z:
            horizon=int(z['horizon']);ext=z['extinction_time'];check('censoring '+path.name,np.array_equal(z['censored'],ext>horizon))
            for j,sid in enumerate(z['series_ids']):
                index=raw[str(sid)]['series_index'];expected=np.random.default_rng(415150+index).integers(0,len(raw[str(sid)]['values']),64)
                check('phases '+str(sid),np.array_equal(expected,z['phases'][j*64:(j+1)*64]))
                for label,t in [('G23',69),('G50',150),('G100',300),('step500',500)]:
                    row=mci[str(sid),2,1,label];observed=int(np.sum(ext[j*64:(j+1)*64]>t))
                    check('MC endpoint '+str(sid)+'/'+label,observed==int(row['survivors']) and t<=horizon)
    # Saved data/protocol provenance, including append-only historical prefixes.
    selection=json.loads((P/'selection.json').read_text())
    for path,wanted in selection['input_hashes'].items():check('input hash '+path,digest(ROOT/path)==wanted)
    text=(P/'protocol.md').read_text();prefixes=set();h=hashlib.sha256()
    for line in text.splitlines(keepends=True):h.update(line.encode());prefixes.add(h.hexdigest())
    for name in ('phase1-validation.json','numerical-validation.json','selection.json','pgf-validation.json'):
        value=json.loads((P/name).read_text());check('historical registration '+name,value['protocol_sha256'] in prefixes)
    for name,field in [('run-manifest.json','hashes'),('completion-manifest.json','input_hashes')]:
        value=json.loads((P/name).read_text())
        for file,wanted in value[field].items():check('run hash '+name+'/'+file,(wanted in prefixes) if file=='protocol.md' else digest(P/file)==wanted)
    for group in selection['overlaps']:
        a,b=group
        for k in (2,3):
            for s in (1,5,20):check('shared control D '+a+f'/{k}/{s}',abs(ci[a,k,s]['D']-ci[b,k,s]['D'])<1e-12)
    ledger=list(csv.DictReader((P/'departures.csv').open()));counts=collections.Counter(r['hypothesis'] for r in ledger)
    check('all departure counts',dict(counts)=={'H1_AR':224,'H1_full':211,'H2':55,'H3_AR':5,'H3_full':5,'MC-validation':1},counts=dict(counts))
    check('report ends correctly',(P/'report.md').read_text().rstrip().endswith('Status: done'))
    # One joined review table makes thresholds, predicted invasion, and persistence comparable.
    joined=[]
    for c in cells:
        r=c.copy()
        for name in ('AR','full'):r['predicted_invasion_'+name]=None if c['prediction_'+name] is None else c['prediction_'+name]<.05
        for label in ('G23','G50','G100','step500'):r['survival_'+label]=float(pi[c['series_id'],c['stages'],c['stay'],label]['survival'])
        joined.append(r)
    write_csv(P/'results-by-series.csv',joined)
    result=dict(date=date(),seconds=time.monotonic()-start,passed=all(r['passed'] for r in checks),count=len(checks),
        passed_count=sum(r['passed'] for r in checks),max_independent_root_error=maxroot,max_independent_Fourier_error=maxfreq,
        checks=checks,engine_hashes={f.name:digest(f) for f in (P.parent/'world').glob('*.py')},
        report_sha256=digest(P/'report.md'),note='Artifact verification does not relabel the one registered Monte Carlo diagnostic failure.')
    write_json(dest,result)
    print(result['passed_count'],'/',len(checks),'verification checks passed')
    for r in checks:
        if not r['passed']:print(r)
    if not result['passed']:sys.exit(1)
