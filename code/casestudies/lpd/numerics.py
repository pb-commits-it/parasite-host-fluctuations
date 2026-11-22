"""Use the project's engine; accelerate only its verified product recurrence locally."""
import os,sys,ctypes,subprocess,types,hashlib,math,json
from pathlib import Path
sys.dont_write_bytecode=True
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0,str(HERE))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import engine_surfaces as es
import helpers_continued as h
from study_io import read,save,RESULTS,now
eng=es.eng
libpath=HERE/'products.so'
if not libpath.exists() or libpath.stat().st_mtime < (HERE/'products.c').stat().st_mtime:
    subprocess.run(['cc','-O3','-shared','-fPIC',str(HERE/'products.c'),'-o',str(libpath)],check=True)
lib=ctypes.CDLL(str(libpath));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.products.argtypes=[ctypes.c_int]*5+[ptr]*8;lib.products.restype=ctypes.c_int

def fast_products(self,rates,y,burn,pairs):
    if self.period>1:raise ValueError('Accelerator supports primitive models only')
    rates=np.ascontiguousarray(rates);T,R,K=rates.shape;n=self.n
    ey=math.exp(y);A=np.ascontiguousarray(self.base_parts[0]+ey*self.base_parts[1]);Ay=np.ascontiguousarray(ey*self.base_parts[1]);P=np.ascontiguousarray(self.rate_parts[0]+ey*self.rate_parts[1]);Py=np.ascontiguousarray(ey*self.rate_parts[1]);v=np.ascontiguousarray(self.v)
    logs=np.zeros(R);slopes=np.zeros(R)
    if lib.products(T,R,K,n,burn,rates,v,A,Ay,P,Py,logs,slopes):raise ValueError('Product failed')
    if pairs:return (logs[:pairs]+logs[pairs:])/2,(slopes[:pairs]+slopes[pairs:])/2
    return logs,slopes

def direct(mode,b,sigma,phi,seed=2026111523,steps=24000,pairs=12,burn=1600):
    m=eng.Engine(es.make_description(dict(mode=mode,b=b),sigma,phi)).critical()
    m._products=types.MethodType(fast_products,m)
    try:
        return m.loss(steps=steps,burn=burn,pairs=pairs,seed=seed,max_iterations=64)
    except eng.ModelError as exc:
        with (RESULTS/'root_solver_fallbacks.jsonl').open('a') as f:
            f.write(json.dumps(dict(timestamp=now(),mode=mode,b=b,sigma=sigma,phi=phi,seed=seed,error=str(exc)))+'\n')
        from scipy.optimize import brentq
        rates=m.rates(m.environment.sample(steps+burn,pairs,seed,antithetic=True))
        def fun(y):return float(m._products(rates,y,burn,pairs)[0].mean())
        J=float(m.jensen()/m.b);lo=-max(1.,J+1.);hi=max(1.,J+1.)
        for _ in range(10):
            if fun(lo)<0 and fun(hi)>0:break
            lo*=2;hi*=2
        y=float(brentq(fun,lo,hi,xtol=1e-10,rtol=1e-12,maxiter=100))
        r,sl=m._products(rates,y,burn,pairs);residual=abs(float(r.mean()/sl.mean()))
        if residual>=1e-7:raise ValueError('Bracketed root failed residual tolerance')
        samples=y-r/sl.mean()
        return dict(D=y,se=float(samples.std(ddof=1)/math.sqrt(pairs)),J=J,root_residual=residual,root_method='brentq_same_engine_forcing')

class Surfaces:
    def __init__(self,cache_name='direct_engine_predictions.json'):
        self.data=read(RESULTS/'engine_surfaces.json')['surfaces'];self.interp={};self.raw={};self.se={};self.j={}
        self.cachepath=RESULTS/cache_name;self.cache=read(self.cachepath) if self.cachepath.exists() else {}
        for k,s in self.data.items():
            sig=np.array(s['sigmas']);phi=np.array(s['phis']);D=np.array(s['D']);v=np.zeros_like(D);v[1:]=D[1:]/sig[1:,None]**2
            # Analytic limit at zero for every surface comes from its own stored D2.
            v[0]=np.array(s['D2'])[1]/sig[1]**2
            self.interp[k]=RegularGridInterpolator((sig,phi),v,bounds_error=True)
            self.raw[k]=RegularGridInterpolator((sig,phi),D,bounds_error=True)
            self.se[k]=RegularGridInterpolator((sig,phi),np.array(s['se']),bounds_error=True)
            self.j[k]=RegularGridInterpolator((sig,phi),np.array(s['J']),bounds_error=True)
    def eval(self,key,sigma,phi,direct_outside=True):
        s=self.data[key]
        if sigma==0:return dict(D=0.,se=0.,D_raw_linear=0.,method='zero',J=0.)
        if min(s['sigmas'])<=sigma<=max(s['sigmas']) and min(s['phis'])<=phi<=max(s['phis']):
            return dict(D=float(self.interp[key]([[sigma,phi]])[0])*sigma*sigma,se=float(self.se[key]([[sigma,phi]])[0]),D_raw_linear=float(self.raw[key]([[sigma,phi]])[0]),method='surface_D_over_sigma2',J=float(self.j[key]([[sigma,phi]])[0]))
        if not direct_outside:return dict(D=None,se=None,D_raw_linear=None,method='outside_sensitivity_grid',J=None)
        ck=f'{key}:{sigma:.12g}:{phi:.12g}'
        if ck not in self.cache:
            d=direct(s['mode'],s['b'],sigma,phi)
            self.cache[ck]=dict(D=d['D'],se=d['se'],D_raw_linear=d['D'],method='direct_engine',J=d['J'],root_residual=d['root_residual'])
            if len(self.cache)%5==0:self.flush()
        return self.cache[ck]
    def flush(self):save(self.cachepath,self.cache)

def validate():
    rng=np.random.default_rng(2026111520);checks=[]
    def ck(name,err,tol,count):
        r=dict(check=name,error=float(err),tolerance=tol,cases=count,passed=bool(err<tol));checks.append(r);print(r,flush=True)
    errors=[]
    for mode,b in [('primary_coarse',7.5),('primary_fine',2.3),('alt2_coarse',24),('patches4_coarse',7.5),('contact_fine',2.3),('chain4_fine',24)]:
        m=eng.Engine(es.make_description(dict(mode=mode,b=b),.3,.4)).critical();rates=m.rates(m.environment.sample(700,3,121,antithetic=True))
        r,s=m._products(rates,.04,50,3);r2,s2=fast_products(m,rates,.04,50,3);errors.extend(np.abs(np.r_[r-r2,s-s2]))
    ck('C products vs original engine products',max(errors),1e-10,6)
    errors=[];acf_err=[];thermal=[]
    for _ in range(50):
        a=rng.uniform(1.1,10);b=rng.uniform(1.1,40);sig=rng.uniform(.1,1.5);phi=rng.uniform(-.8,.9)
        d=eng.Description([eng.Stage('a',a),eng.Stage('b',b)],[eng.Transfer('a','b',driver='h'),eng.Transfer('b','a',birth=True)],[eng.Driver('h',sig,phi)])
        errors.append(abs(eng.Engine(d).coefficient()['D2']-h.gamma0_two_stage(a,b,sig,phi)))
        acf_err.append(abs(h.gamma0_from_full_acf(a,b,sig,{j:phi**j for j in range(1,401)},False)-h.gamma0_two_stage(a,b,sig,phi)))
        thermal.append(abs(h.thermal_loss_d2_one_exposed(a,b,sig,phi,.05,.08)-h.thermal_loss_d2_resolvent(a,b,sig,phi,.05,.08)))
    ck('Ring formula vs engine',max(errors),1e-9,50);ck('ACF lag sum vs AR1 formula',max(acf_err),1e-9,50);ck('Thermal formula vs independent matrix resolvent',max(thermal),1e-9,50)
    t=np.arange(31.);y=rng.normal(size=31)
    q=h.series_block_bootstrap_se(t,y,np.random.default_rng(10));r=h.series_block_bootstrap_se(t,y,np.random.default_rng(10),force_loop=True)
    ck('Bootstrap vectorized vs loop',max(abs(q[k]-r[k]) for k in q),1e-10,200)
    t=np.arange(30.);X=np.column_stack([np.ones(30),t,rng.normal(size=(30,2))]);Q=h.orthonormal_basis_from_design(X)
    yy=rng.normal(size=(30000,30));ee=yy-(yy@Q)@Q.T;rr=(ee[:,:-1]*ee[:,1:]).mean(axis=1)/(ee*ee).mean(axis=1)
    ck('Projection white-noise mean',abs(rr.mean()-h.projected_white_noise_r1_null_mean(t,Q)),.01,30000)
    surface=Surfaces();points=[]
    for j in range(12):
        b=[24.,7.5,2.3][j%3];name=['long','moderate','short'][j%3];sig=[.025,.2,.7,1.3][j//3];phi=[-.4,.08,.42,.83][j//3]
        d=direct('primary_coarse',b,sig,phi,seed=2026111522+j,steps=30000,pairs=12)
        v=surface.eval('primary_coarse_'+name,sig,phi);tol=4*math.hypot(d['se'],v['se'])+max(.002,.05*abs(d['D']))
        points.append(dict(cycle=name,sigma=sig,phi=phi,direct=d['D'],direct_se=d['se'],interpolated=v['D'],raw_linear=v['D_raw_linear'],tolerance=tol,error=abs(d['D']-v['D'])))
        print('interpolation check',j,points[-1],flush=True)
    ck('Amplitude-aware interpolation',max(r['error']/r['tolerance'] for r in points),1.,12)
    save(RESULTS/'continuation_checks.json',dict(timestamp=now(),checks=checks,interpolation=points))
    assert all(c['passed'] for c in checks),'Validation failed; do not interpret empirical results'
if __name__=='__main__':validate()
