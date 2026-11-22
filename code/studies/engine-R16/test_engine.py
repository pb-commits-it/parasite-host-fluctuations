"""Locked criteria, declared before the first run. python3 -I test_engine.py.

Algebra: 60 random cases per family, relative error <= 1e-9.
Products: two cases per family, |D-D2| <= 4 SE + .05 |D2|.
Bound: D <= J + 4 SE + 1e-7 for every product run.
Paths: five cases, |C-J2+V/2| <= 4 SE(V)/2 + .01 |J2|.
Poisson extinction: fixed-point difference <= 4 binomial SE.
Failed outputs are retained before rerunning; tolerances never auto-adjust.
"""
from pathlib import Path
import importlib.util
import sys, time, json
import numpy as np

HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec=importlib.util.spec_from_file_location('lifecycle_engine',HERE/'lifecycle_engine.py')
e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
Stage,Transfer,Driver,Patches,Description,Engine=e.Stage,e.Transfer,e.Driver,e.Patches,e.Description,e.Engine
ALGEBRA_RTOL=1e-9
SMALL_NOISE_REL=0.05
SIGMA=0.04
N_RANDOM=60
N_PAIRS=24
N_STEPS=24000
V_ENVIRONMENTS=80
V_LINES=300
V_STEPS=1600
FAMILIES=['ring','contact','chain','fixed','independent_sequence','shared_host','parallel_no_memory','parallel_memory','optional','bypass','island']


def ring(taus,phi=.5,sigma=SIGMA,shape=None,contact=None,drivers_on=None,patches=None):
    stages=[Stage(str(i),float(t)) for i,t in enumerate(taus)]
    if shape:
        j,kind,m=shape;stages[j]=Stage(str(j),float(taus[j]),kind,m)
    if contact is not None:
        stages[0]=Stage('0',float(taus[0]),death=contact/taus[0],contact=True)
    if drivers_on is None:drivers_on={0:('h',1.,phi)}
    drivers={name:Driver(name,sigma,ph,'bounded' if contact is not None else 'lognormal') for name,_,ph in drivers_on.values()}
    routes=[]
    for i in range(len(taus)):
        name,eta,_=drivers_on.get(i,(None,1.,0))
        routes.append(Transfer(str(i),str((i+1)%len(taus)),driver=name,elasticity=eta,birth=i==len(taus)-1))
    return Engine(Description(stages,routes,list(drivers.values()),patches=patches or Patches())).critical()


def transforms(taus,phi):
    t=np.asarray(taus,float)
    return phi/(phi+(1-phi)*t)


def kg(taus,phi):
    R=np.prod(transforms(taus,phi));T=sum(taus)
    return (1+phi)/(2*(1-phi)*T)-R/(1-R)


def case(family,rng,sigma=SIGMA,example=None):
    tau=rng.uniform(1.5,12,3);phi=float(rng.uniform(.1,.85))
    if example is not None:
        tau=np.array([3.,6.,4.])+example;phi=.35+.25*example
    if family=='ring':
        engine=ring(tau,phi,sigma);K=kg(tau,phi)
    elif family=='contact':
        d=float(rng.uniform(.15,.9)) if example is None else .4+.3*example
        engine=ring(tau,phi,sigma,contact=d)
        q=transforms(tau,phi);R=np.prod(q)
        K=d*d*kg(tau,phi)+d*(1-d)*(q[0]-R)/(1-R)
    elif family in ('chain','fixed'):
        m=int(rng.integers(2,6));tau[1]=m*rng.uniform(1.2,4)
        if family=='fixed':tau[1]=float(rng.integers(2,18))
        engine=ring(tau,phi,sigma,shape=(1,family,m))
        q=transforms(tau,phi)
        q[1]=(phi/(phi+(1-phi)*tau[1]/m))**m if family=='chain' else phi**int(tau[1])
        R=np.prod(q);K=(1+phi)/(2*(1-phi)*sum(tau))-R/(1-R)
    elif family=='independent_sequence':
        ph2=float(rng.uniform(.05,.8))
        engine=ring(tau,phi,sigma,drivers_on={0:('a',1.,phi),1:('b',1.,ph2)})
        K=kg(tau,phi)+kg(tau,ph2)
    elif family=='shared_host':
        aa=rng.uniform(.4,1.8,2)*rng.choice([-1.,1.],2)
        if example is not None:aa=np.array([1.,1. if example==0 else -.7])
        engine=ring(tau,phi,sigma,drivers_on={0:('h',aa[0],phi),1:('h',aa[1],phi)})
        q=transforms(tau,phi);R=np.prod(q);total=0.
        for i in range(2):
            for j in range(2):
                trans=1.;k=(j+1)%len(tau)
                while True:
                    trans*=q[k]
                    if k==i:break
                    k=(k+1)%len(tau)
                total+=aa[i]*aa[j]*trans
        K=(1+phi)/(2*(1-phi)*sum(tau))*sum(aa)**2-total/(1-R)
    elif family.startswith('parallel'):
        if family=='parallel_no_memory':phi=0.
        w=float(rng.uniform(.15,.85));rho=float(rng.uniform(-.6,.8));eta=rng.uniform(.5,1.5,2)
        Rho=np.array([[1.,rho],[rho,1.]])
        drivers=[Driver('a',sigma,phi),Driver('b',sigma,phi)]
        transfers=[Transfer('0','1',w,driver='a',elasticity=eta[0]),Transfer('0','2',1-w,driver='b',elasticity=eta[1]),Transfer('1','0',birth=True),Transfer('2','0',birth=True)]
        engine=Engine(Description([Stage(str(i),t) for i,t in enumerate(tau)],transfers,drivers,Rho)).critical()
        shares=np.array([w,1-w]);u=shares*eta;V=u@Rho@u;G=tau[0]+shares@tau[1:]
        q=transforms(tau,phi);R=q[0]*(shares@q[1:])
        K=(1+phi)/(2*(1-phi))*V/G-q[0]/(1-R)*np.sum(u*q[1:]*(Rho@u))
    elif family=='optional':
        f=float(rng.uniform(.1,.9));survival=float(rng.uniform(.3,1));Y=1-f+f*survival;alpha=f*survival/Y
        transfers=[Transfer('0','1',driver='h'),Transfer('1','0',1-f,birth=True),Transfer('1','2',f),Transfer('2','0',yield_=survival,birth=True)]
        engine=Engine(Description([Stage(str(i),t) for i,t in enumerate(tau)],transfers,[Driver('h',sigma,phi)])).critical()
        q=transforms(tau,phi);R=q[0]*q[1]*(1-alpha+alpha*q[2]);G=tau[0]+tau[1]+alpha*tau[2]
        K=(1+phi)/(2*(1-phi)*G)-R/(1-R)
    elif family=='bypass':
        f=float(rng.uniform(.1,.9));survival=float(rng.uniform(.3,1));Y=1-f+f*survival;alpha=(1-f)/Y
        transfers=[Transfer('0','1',1-f,driver='h'),Transfer('0','0',f,yield_=survival,birth=True),Transfer('1','0',birth=True)]
        engine=Engine(Description([Stage(str(i),t) for i,t in enumerate(tau[:2])],transfers,[Driver('h',sigma,phi)])).critical()
        q=transforms(tau[:2],phi);R=q[0]*(1-alpha+alpha*q[1]);G=tau[0]+alpha*tau[1]
        K=alpha**2*((1+phi)/(2*(1-phi)*G)-q[0]*q[1]/(1-R))
    elif family=='island':
        P=int(rng.integers(2,6));m=float(rng.uniform(.08,.7));rho=float(rng.uniform(0,.8));ell=float(rng.uniform(0,.25))
        patch=Patches(P,rho,'1',m,'island',ell)
        engine=ring(tau,phi,sigma,patches=patch)
        a=1-m*ell;b=1-m;s=1-1/tau;u=s.copy();u[1]*=a;t=1/(1-u);T=sum(t)
        Ra=np.prod(phi*(1-u)/(1-phi*u));Rb=Ra*(b/a)*(1-phi*a*s[1])/(1-phi*b*s[1]);q=rho+(1-rho)/P
        S=(1+phi)/(2*(1-phi)*T)-Ra/(1-Ra);H=Rb/(1-Rb);K=q*S-(1-q)*H
    else:raise ValueError(family)
    return engine,float(K*sigma*sigma)


class Tee:
    def __init__(self,file):self.file=file
    def write(self,s):sys.__stdout__.write(s);self.file.write(s);self.file.flush()
    def flush(self):self.file.flush()


def main():
    output=HERE/'test_engine_output.txt'
    if output.exists():
        n=1
        while (HERE/f'test_engine_output_attempt{n}.txt').exists():n+=1
        suffix='verification' if 'ALL ' in output.read_text() and 'CHECKS PASSED' in output.read_text() else 'attempt'
        while (HERE/f'test_engine_output_{suffix}{n}.txt').exists():n+=1
        output.rename(HERE/f'test_engine_output_{suffix}{n}.txt')
    with output.open('w') as f:
        sys.stdout=Tee(f)
        start=time.monotonic();results=[];details=[]
        print('Locked tolerances:',ALGEBRA_RTOL,SMALL_NOISE_REL,'four standard errors; path allowance .01 J2')
        print('Random parameter sets per formula:',N_RANDOM,'seed:',481291)
        rng=np.random.default_rng(481291)
        def check(name,ok,detail):
            print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}",flush=True)
            results.append({'name':name,'pass':bool(ok),'detail':detail})
            if not ok:raise AssertionError(name)
        for family in FAMILIES:
            worst=0.
            for _ in range(N_RANDOM):
                model,pred=case(family,rng)
                actual=model.coefficient()['D2']
                worst=max(worst,abs(actual-pred)/max(abs(pred),1e-300))
            check(f'algebra {family}',worst<=ALGEBRA_RTOL,f'{N_RANDOM} sets; max relative error {worst:.4g}')
        # Guard tests are constructive domain checks, separate from formulas.
        periodic=ring([1.,1.],.5,SIGMA)
        refused=False
        try:periodic.coefficient()
        except e.PeriodicModelError:refused=True
        check('periodic guard',refused,f'period {periodic.period}; no coefficient')
        for label,build in [('unbounded contact',lambda:Engine(Description([Stage('x',4,death=.1,contact=True)],[Transfer('x','x',driver='h',birth=True)],[Driver('h',.1,.5)]))),
                            ('no contact staying',lambda:ring([1.,4.],contact=.5)),
                            ('short chain',lambda:ring([2.,2.3],shape=(1,'chain',4)))]:
            refused=False
            try:build()
            except e.ModelError:refused=True
            check(label,refused,'invalid data refused')
        examples=[]
        for k,family in enumerate(FAMILIES):
            for j in range(2):
                model,pred=case(family,np.random.default_rng(900+k*2+j),example=j)
                ans=model.loss(steps=N_STEPS,burn=1500,pairs=N_PAIRS,seed=3000+k*2+j)
                check(f'products {family} {j+1}',abs(ans['D']-pred)<=4*ans['se']+SMALL_NOISE_REL*abs(pred),
                      f"D {ans['D']:.9g} SE {ans['se']:.3g}; D2 {pred:.9g}")
                check(f'bound {family} {j+1}',ans['D']<=ans['J']+4*ans['se']+1e-7,f"D {ans['D']:.9g} J {ans['J']:.9g}")
                details.append({'family':family,'case':j,'prediction':pred,**ans})
                if j==0:examples.append((family,model))
        selected=['ring','contact','independent_sequence','island','parallel_memory']
        for k,name in enumerate(selected):
            model=dict(examples)[name];c=model.coefficient()
            v=model.lineage_variance(V_STEPS,V_ENVIRONMENTS,V_LINES,seed=712+k)
            delta=abs(c['C']-c['J2']+v['V']/2)
            check(f'lineages {name}',delta<=4*v['se']/2+.01*abs(c['J2']),
                  f"C {c['C']:.9g} J2 {c['J2']:.9g} V {v['V']:.9g} SE(V) {v['se']:.3g}")
        v=periodic.lineage_variance(200,10,50,seed=185)
        check('periodic single cohort',abs(v['V'])<1e-14,f"V {v['V']:.3g}")
        deterministic=ring([1.,1.,1.],.6,.1)
        exact=deterministic.loss(steps=9000,burn=300,pairs=8,seed=718)
        check('periodic matrix products',abs(exact['D']-.1**2/2)<1e-10,
              f"D {exact['D']:.12g}; bound {exact['J']:.12g}; one cohort")
        lossless=ring([4.,6.],.7,.5,contact=0.)
        no_death=lossless.loss(steps=3000,burn=300,pairs=8,seed=719)
        check('no-death contact conservation',abs(no_death['D'])<1e-12,
              f"D {no_death['D']:.12g}; every matrix conserves reproductive value")
        # Independent scalar equation: q=s q+(1-s) exp(mu(q-1)).
        mu=1.35;s=.6;q=0.
        for _ in range(10000):
            qnew=s*q+(1-s)*np.exp(mu*(q-1))
            if abs(qnew-q)<1e-14:break
            q=qnew
        pop=Engine(Description([Stage('x',1/(1-s))],[Transfer('x','x',yield_=mu,birth=True)]))
        fixed=pop.offspring_fixed_point()[0]
        check('Poisson fixed point',abs(fixed-q)<1e-10,f'q {q:.12g}; engine {fixed:.12g}')
        obs=pop.population([1],100,trials=6000,seed=2026)
        check('Poisson extinction',abs(obs['fraction']-q)<=4*obs['se'],f"fraction {obs['fraction']:.8g} SE {obs['se']:.4g} fixed point {q:.8g}")
        endpoint=pop.extinction([1],100,trials=6000,seed=2027,constant=True)
        check('Poisson endpoint sampler',abs(endpoint['fraction']-q)<=4*endpoint['se'],f"fraction {endpoint['fraction']:.8g} SE {endpoint['se']:.4g}")
        # One-step means distinguish event sampling from Poissonizing stays.
        for law in ['poisson','negative_binomial','shared_groups']:
            m=ring([3.,5.],sigma=0)
            obs=m.population([100,70],1,trials=20000,seed=82,offspring=law)
            average=obs['counts'].mean(axis=0);se=obs['counts'].std(axis=0,ddof=1)/np.sqrt(20000)
            target=m.mean_matrix@np.array([100,70])
            check('whole-number mean '+law,np.all(abs(average-target)<=5*se+1e-9),f'mean {average.tolist()} matrix {target.tolist()}')
        # Cross-lag orientation: unequal AR memories, nonzero correlation and
        # signed elasticities. Compare the resolvent to an explicit lag sum.
        correlated=Engine(Description([Stage('a',3),Stage('b',5)],
            [Transfer('a','b',driver='x',elasticity=-.7),Transfer('b','a',driver='z',birth=True)],
            [Driver('x',.1,.2),Driver('z',.2,.7)],np.array([[1,.3],[.3,1]]))).critical()
        B=correlated.sensitivities();Q=np.eye(2)-np.outer(correlated.v,correlated.w)
        T=correlated.mean_matrix@Q;c=np.einsum('i,dij,j->d',correlated.w,B,correlated.v)
        lag=c@correlated.environment.corr@c/2;power=np.eye(2)
        for mlag in range(300):
            for i in range(2):
                for j in range(2):
                    lag-=correlated.environment.phi[i]**(mlag+1)*correlated.environment.corr[i,j]*float(correlated.w@B[i]@power@Q@B[j]@correlated.v)
            power=power@T
        check('signed unequal-memory cross-lags',abs(lag-correlated.coefficient()['C'])<=1e-12,f'lag sum {lag:.12g}')
        binary=e.Environment([Driver('a',.2,.6,'bounded',.3),Driver('b',.2,.6,'bounded',.3)],np.array([[1,.4],[.4,1]]))
        z=binary.sample(10000,64,451,standardized=True)
        empirical=float(np.mean(z[:,:,0]*z[:,:,1]));lagged=float(np.mean(z[1:,:,0]*z[:-1,:,1]))
        check('correlated bounded law',abs(empirical-.4)<.03 and abs(lagged-.24)<.03,f'lag zero {empirical:.6g}; lag one {lagged:.6g}')
        contact=ring([4.,6.],contact=.5)
        check('constant contact PGF',np.allclose(contact.pgf_step(np.ones(contact.n)),1),'PGF(1)=1 with arithmetic mean rates')
        # Stable survival recursion is needed when a bottleneck is followed
        # by favourable conditions in the backward composition.
        tiny=pop._pgf_survival(np.array([1e-25]))[0]
        check('tiny survival retained',tiny>0 and abs(tiny/1e-25-(s+(1-s)*mu))<1e-12,f'survival {tiny:.12g}')
        together=Engine(Description([Stage('x',4,death=.125,contact=True),Stage('y',12,'chain',4,.01)],
            [Transfer('x','y',.5,driver='a'),Transfer('x','y',.5,driver='b'),Transfer('y','x',birth=True)],
            [Driver('a',.2,.5,'bounded'),Driver('b',.2,.5,'bounded')],
            patches=Patches(2,0.,'y',.15,'nearest',.1))).critical()
        initial=np.arange(together.n)*10+40
        obs=together.population(initial,1,trials=20000,seed=458)
        avg=obs['counts'].mean(axis=0);se=obs['counts'].std(axis=0,ddof=1)/np.sqrt(20000)
        target=together.mean_matrix@initial
        check('combined whole-number mean',np.all(abs(avg-target)<=5*se+1e-9),
              f'max standardized error {np.max(abs(avg-target)/se):.6g}; contact, chain, routes, death, movement, transit loss')
        raw=Engine(correlated.description,birth_scale=1.4)
        check('threshold versus growth coefficient',raw.coefficient(threshold=False)['D2'] is None and
              abs(raw.coefficient()['D2']-raw.critical().coefficient()['D2'])<1e-13,
              'raw growth correction is not mislabeled as a threshold loss')
        elapsed=time.monotonic()-start
        print(f'ALL {len(results)} CHECKS PASSED; wall seconds {elapsed:.3f}')
        (HERE/'test_engine_results.json').write_text(json.dumps({'checks':results,'products':details,'wall_seconds':elapsed},indent=2))
        sys.stdout=sys.__stdout__

if __name__=='__main__':main()
