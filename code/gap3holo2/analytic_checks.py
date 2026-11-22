"""Independent numerical checks of the analytic cores stated in gap3holo2.md.
Registered tolerance: identities 1e-30, derivative identity 1e-25,
order-12 threshold at p=200 1e-15. Added before first execution:
Laplace coefficient construction at orders 2,4,6 checked against the explicit
bound (no fitted constants); complex-contraction integral q<1 at real centers;
pointwise complex two-step metric inequality slack 1e-25.
"""
import importlib.util, json, math, time
from pathlib import Path
import mpmath as mp
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec=importlib.util.spec_from_file_location('own_checks',P/'checks.py'); core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
mp.mp.dps=75

def tricomi_rate(s,c,p):
    k=s*s*p/c
    return mp.log(s)+mp.diff(lambda b:mp.hyperu(p,b,k),1)/mp.hyperu(p,1,k)

def density(s,c,p,which='rate'):
    K=s*s/c; t0=(mp.sqrt(1+4/K)-1)/2; x0=mp.log(t0)
    phase=lambda x:K*mp.exp(x)+mp.log1p(mp.exp(-x))
    f0=phase(x0); curvature=K*t0+t0/(1+t0)**2
    scale=1/mp.sqrt(p*curvature)
    def weight(x): return mp.exp(-p*(phase(x)-f0))
    # Finite endpoints are chosen by exact convex tails, each >220 log units
    # below peak. Quadrature is not interval certification.
    left=x0-8*scale;right=x0+8*scale
    while p*(phase(left)-f0)<240:left-=8*scale
    while p*(phase(right)-f0)<240:right+=8*scale
    pts=[left,x0-2*scale,x0,x0+2*scale,right]
    den=mp.quad(weight,pts)
    g=lambda x:mp.log(s*(1+mp.exp(x)))
    num=mp.quad(lambda x:g(x)*weight(x),pts)
    if which=='J':return den/mp.sqrt(mp.pi/p),num/mp.sqrt(mp.pi/p)
    mean_t=mp.quad(lambda x:mp.exp(x)*weight(x),pts)/den
    cov=mp.quad(lambda x:mp.exp(x)*g(x)*weight(x),pts)/den-mean_t*num/den
    return num/den,K*p*cov

def power(poly, exponent, N):
    # Exact finite power-series algorithm over arbitrary-precision scalars.
    h=[x/poly[0] for x in poly[:N+1]]
    log=core.log_series(h,N);e=core.exp_series([exponent*x for x in log],N)
    return [poly[0]**exponent*x for x in e]

def laplace_coeffs(s,c,N,growth):
    K=s*s/c;t=(mp.sqrt(1+4/K)-1)/2
    M=2*(N-1)
    H=[(-1)**l*(t**(-l-2)-(1+t)**(-l-2))/(l+2) for l in range(M+1)]
    f=[(-1)**j/t**(j+1) for j in range(M+1)]
    if growth:
        g=[mp.log(s*(1+t))]+[(-1)**(j+1)/(j*(1+t)**j) for j in range(1,M+1)]
        f=[sum(g[j]*f[k-j] for j in range(k+1)) for k in range(M+1)]
    b=[]
    for j in range(N):
        m=2*j; q=power(H,-mp.mpf(m+1)/2,m)
        b.append(sum(f[l]*q[m-l] for l in range(m+1)))
    return b

def laplace_constants(s,c,growth):
    K=s*s/c;t=(mp.sqrt(1+4/K)-1)/2;d=t/8
    # Bounds for Phi'' on |z-t|<=d: argument at most 5 asin(1/8).
    low=mp.cos(5*mp.asin(mp.mpf(1)/8))*(1+2*(t-d))/((t+d)**2*(1+t+d)**2)
    high=(1+2*(t+d))/((t-d)**2*(1+t-d)**2)
    rw=d*mp.sqrt(low/2);R=(rw/2)**2;b=R/2
    G=abs(mp.log(s*(1+t)))-mp.log(1-d/(1+t)) if growth else mp.mpf(1)
    M=2*d*G/(rw*(t-d))
    phi0=K*t+mp.log((1+t)/t)
    dm=1/(1+t-d)-K*(t-d)
    dp=K-1/((t+d)*(1+t+d))
    leftG=abs(mp.log(s))+mp.log(1+t) if growth else mp.mpf(1)
    rightG=abs(mp.log(s))+phi0/K+1/K if growth else mp.mpf(1)
    localG=abs(mp.log(s))+mp.log(1+t+d) if growth else mp.mpf(1)
    C=max(2*leftG/dm,2*rightG/(dp*(t+d)),mp.sqrt(2*high)*localG/(low*(t-d)))
    E=4*M+C*(1+b*b)
    L=2*d*mp.exp(-high*d*d/2)/(mp.sqrt(mp.pi)*(t+d))
    return b,E,L,low,high

def main():
    start=time.time();results={};worst=mp.mpf(0);slope_min=mp.inf
    for sv in ['0.3','0.6','0.9']:
        s=mp.mpf(sv);a=1-s
        for p in [2,20]:
            q,dy=density(s,a,p);u=tricomi_rate(s,a,p)
            err=abs(q-u);worst=max(worst,err);slope_min=min(slope_min,dy)
            assert err<mp.mpf('1e-30');assert dy>0
        p=200
        if sv=='0.9':
            exact=mp.mpf('0.00023')
            for _ in range(7):
                value,slope=density(s,a*mp.exp(exact),p)
                exact-=value/slope
        else:
            exact=mp.findroot(lambda y:tricomi_rate(s,a*mp.exp(y),p),(0,mp.mpf('.01')))
        co=core.gamma_stein(sv,12,160)
        approx=sum(mp.mpf(str(x))/p**(k+1) for k,x in enumerate(co))
        err=abs(exact-approx);assert err<mp.mpf('1e-15')
        results[sv]={'threshold_p200':str(exact),'series12_error':str(err)}
        print('PASS exact gamma',sv,'D(p200)',mp.nstr(exact,20),'N12 error',mp.nstr(err,5),flush=True)
    print('PASS density vs Tricomi 6 cases max',mp.nstr(worst,6),'positive slope minimum',mp.nstr(slope_min,6),flush=True)
    # Lower bank of cut, approached with an explicit tiny negative imaginary part.
    worst=mp.mpf(0);worst_real=mp.mpf(0);worst_ibp=mp.mpf(0)
    for p in [mp.mpf('-2.5'),mp.mpf('-5.5'),mp.mpf('-10.5')]:
        for x in map(mp.mpf,['0.7','2.5','10']):
            z=mp.mpc(-x,-mp.mpf('1e-65'))
            for b in map(mp.mpf,['0.9','1','1.1']):
                rhs=mp.gamma(b-p)*mp.exp(-x)*(mp.exp(mp.j*mp.pi*(b-1))*mp.hyperu(b-p,b,x)/mp.gamma(p)+mp.exp(mp.j*mp.pi*p)*mp.hyp1f1(b-p,b,x)/mp.gamma(b))
                lhs=mp.hyperu(p,b,z)
                err=abs(lhs-rhs)/max(1,abs(lhs));worst=max(worst,err)
                assert err<mp.mpf('1e-30')
            q=1-p
            # Integral moments via distinct U parameters, plus direct log derivative.
            norm=mp.gamma(q)*mp.hyperu(q,1,x)
            mt=mp.gamma(q+1)*mp.hyperu(q+1,2,x)/norm
            mt2=mp.gamma(q+2)*mp.hyperu(q+2,3,x)/norm
            ibp=abs(x*mt2+(x-1)*mt-q);worst_ibp=max(worst_ibp,ibp);assert ibp<mp.mpf('1e-30')
            elog=mp.diff(lambda b:mp.log(mp.gamma(b-p)*mp.hyperu(b-p,b,x)),1)
            A0=mp.exp(-x)*norm/mp.gamma(p)
            lhs=mp.re(mp.diff(lambda b:mp.hyperu(p,b,z),1))/A0
            err=abs(lhs-elog);worst_real=max(worst_real,err);assert err<mp.mpf('1e-25')
            upper=2*q/(x-1+mp.sqrt((x-1)**2+4*x*q))
            assert elog<mp.log(upper)
    print('PASS connection 27 cases',mp.nstr(worst,5),'real-part derivative 9 cases',mp.nstr(worst_real,5),'IBP',mp.nstr(worst_ibp,5),flush=True)
    # The limiting sign check is not substituted for its proof.
    for sv in ['0.3','0.6','0.9']:
        s=mp.mpf(sv);q=mp.mpf('100001.5');x=(q-1)*s*s/(1-s)
        upper=2*q/(x-1+mp.sqrt((x-1)**2+4*x*q))
        value=mp.log(s*upper)
        assert value<0
        print('PASS contradiction sign',sv,mp.nstr(value,10),'limit',mp.nstr(mp.log(1-s),10),flush=True)
    # Explicit Watson/Laplace error constants: check entire integral, not fitted errors.
    largest_ratio=mp.mpf(0)
    for sv in ['0.3','0.6','0.9']:
        s=mp.mpf(sv);c=(1-s)*mp.exp(mp.mpf('.02'))
        for p in [20,200]:
            values=density(s,c,p,'J')
            for growth in [False,True]:
                b,E,L,lo,hi=laplace_constants(s,c,growth)
                assert lo>0 and hi>lo and values[0]>L
                for N in [2,4,6]:
                    coeff=laplace_coeffs(s,c,N,growth)
                    approx=sum(coeff[j]*mp.gamma(j+mp.mpf('.5'))/mp.sqrt(mp.pi)/p**j for j in range(N))
                    error=abs(values[int(growth)]-approx)
                    bound=E*mp.factorial(N)/(b*p)**N
                    largest_ratio=max(largest_ratio,error/bound)
                    assert error<=bound
    print('PASS explicit Laplace remainder 36 cases, max error/bound',mp.nstr(largest_ratio,6),flush=True)
    # Complex Mobius hyperbolic derivative bound, independent direct derivative.
    worst_slack=-mp.inf;count=0
    for u in map(mp.mpf,['.1','.6','1.2']):
      for v in map(mp.mpf,['-.8','0','.8']):
       A=mp.mpc(u,v)
       for b0 in map(mp.mpf,['.2','2']):
        for b1 in map(mp.mpf,['.1','1','10']):
         bound=(mp.sqrt(u*u+b1)-u)/(mp.sqrt(u*u+b1)+u)
         for wr in map(mp.mpf,['.01','.2','2','20']):
          for wi in map(mp.mpf,['-3','0','3']):
           w=mp.mpc(wr,wi);t=A+b0/w;g=A+b1/t
           measured=abs(b0*b1/(w*w*t*t))*wr/mp.re(g)
           worst_slack=max(worst_slack,measured-bound);count+=1
           assert measured<=bound+mp.mpf('1e-25')
    print('PASS complex contraction',count,'points, max signed slack',mp.nstr(worst_slack,5),flush=True)
    results['laplace_max_ratio']=str(largest_ratio);results['connection_max_error']=str(worst)
    (P/'analytic_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PASS analytic checks seconds',round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
