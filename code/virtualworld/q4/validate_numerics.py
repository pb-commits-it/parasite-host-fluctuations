"""Check periodic D and general covariance predictions before empirical use."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();sys.path.insert(0,str(P))
from numerics import *
from world.invasion import products,threshold,resolvent_coefficient

if __name__=='__main__':
    if (P/'numerical-validation.json').exists(): raise RuntimeError('append-only')
    assert json.loads((P/'phase1-validation.json').read_text())['passed']
    assert '\nOK\n' in (P/'engine-tests.log').read_text()
    start=time.monotonic();rows=[]
    def check(name,actual,expected,limit):
        rows.append(dict(check=name,actual=float(actual),expected=float(expected),error=float(abs(actual-expected)),tolerance=limit,passed=bool(abs(actual-expected)<=limit)))
    z=np.array([.2,-.6,.3,1.,.4,-.2,.7,-.9,-.3,.6,-.8,-.4])
    z-=z.mean()
    for k in (2,3):
        for s in (1,5,20):
            m=model(k,s);a,w=kernel(k,s)
            check(f'constant {k}/{s}',loss(k,s,np.ones(12))[0],0,1e-10)
            for phi in (-.5,0,.6,.9):
                estimate=a-np.dot(w,phi**np.arange(1,4097))
                check(f'kernel/resolvent {k}/{s}/{phi}',estimate,resolvent_coefficient(m,phi),1e-10)
                check(f'kernel/ring {k}/{s}/{phi}',estimate,ring_coefficient(means(k,s),phi),1e-10)
            eps=.001
            dplus=loss(k,s,norm_host(np.exp(eps*z)))[0]
            dminus=loss(k,s,norm_host(np.exp(-eps*z)))[0]
            check(f'finite difference covariance {k}/{s}',(dplus+dminus)/(2*eps**2),predict(k,s,norm_host(np.exp(z)))[1],1e-5)
            h=norm_host(np.exp(.1*z));D,_=loss(k,s,h)
            history=np.tile(h,1024+256)[:,None,None];burn=256*len(h)
            ref=threshold(m,history,burn=burn,control=1/sum(means(k,s)),control_mean=None)
            check(f'periodic/engine threshold {k}/{s}',D,ref['D'],1e-7)
            r=products(m,history,D,burn=burn,control=1/sum(means(k,s)),control_mean=None)
            raw=products(m,history,D,burn=burn)
            check(f'finite centering {k}/{s}',r[0][0],raw[0][0],1e-12)
            check(f'periodic growth {k}/{s}',r[0][0],growth(m,h,D),1e-10)
    result=dict(date=date(),seconds=time.monotonic()-start,passed=all(r['passed'] for r in rows),checks=rows,
        protocol_sha256=digest(P/'protocol.md'),code_sha256=digest(P/'numerics.py'))
    write_json(P/'numerical-validation.json',result)
    print(len(rows),'checks; passed',sum(r['passed'] for r in rows),'seconds',result['seconds'])
    for r in rows:
        if not r['passed']: print(r)
    if not result['passed']:sys.exit(1)
