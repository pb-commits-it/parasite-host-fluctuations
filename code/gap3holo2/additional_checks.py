"""Additional checks, tolerances preregistered in research.md before execution."""
import importlib.util,json
from pathlib import Path
import mpmath as mp
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec=importlib.util.spec_from_file_location('own_analytic',P/'analytic_checks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
mp.mp.dps=60
count=0;slack=mp.inf
for sv in ['.3','.6','.9']:
 s=mp.mpf(sv)
 for shift in map(mp.mpf,['-.05','0','.05']):
  c=(1-s)*mp.exp(shift);K=s*s/c;t=(mp.sqrt(1+4/K)-1)/2;d=t/8
  b,E,L,low,high=m.laplace_constants(s,c,False)
  for i in range(48):
   z=t+d*mp.exp(2*mp.pi*mp.j*i/48)
   second=(2*z+1)/(z*z*(1+z)**2)
   diff=mp.re(second)-low;slack=min(slack,diff);count+=1
   assert diff>=-mp.mpf('1e-25');assert abs(second)<=high+mp.mpf('1e-25')
print('PASS local complex saddle bounds',count,'points minimum slack',mp.nstr(slack,6),flush=True)
exact=json.loads((P/'analytic_results.json').read_text());worst=mp.mpf(0)
for sv in ['0.3','0.6','0.9']:
 s=mp.mpf(sv);v=mp.mpf(1)/200;dexact=mp.mpf(exact[sv]['threshold_p200'])
 co=[mp.mpf(str(x)) for x in m.core.gamma_stein(sv,8,120)]
 cmin=(1-s)*mp.exp(-mp.mpf('.1'));lam=cmin/(6*(2*s*s+cmin))
 for N in [2,4,8]:
  Y=sum(co[j]*v**(j+1) for j in range(N));assert abs(Y)<mp.mpf('.1');c=(1-s)*mp.exp(Y)
  b,E,L,_,_=m.laplace_constants(s,c,True)
  coeff=m.laplace_coeffs(s,c,N+1,True)
  residual=abs(sum(coeff[j]*mp.gamma(j+mp.mpf('.5'))/mp.sqrt(mp.pi)*v**j for j in range(N+1)))
  bound=(residual+E*(v/b)**(N+1)*mp.factorial(N+1))/(lam*L)
  error=abs(dexact-Y);worst=max(worst,error/bound);assert error<=bound
print('PASS threshold certificate 9 cases max error/bound',mp.nstr(worst,5),flush=True)
# Absolute complex density normalization and killed transition contraction.
maxvalue=mp.mpf(0);count=0
for sv in ['.3','.6','.9']:
 s=mp.mpf(sv);c=(1-s)*mp.exp(mp.j*mp.mpf('1e-4'));u=s*mp.cos(mp.arg(c)/2)
 kap=lambda h:(mp.sqrt(u*u+abs(c)*h)-u)/(mp.sqrt(u*u+abs(c)*h)+u)
 for p0 in [2,20]:
  p=p0*(1+mp.j*mp.mpf('1e-4'))
  f=lambda h:abs(mp.exp(p*mp.log(p)-mp.loggamma(p)+(p-1)*mp.log(h)-p*h))
  L=mp.quad(f,[0,1,mp.inf]);Q=mp.quad(lambda h:f(h)*kap(h),[0,1,mp.inf])/L
  for phi in [0,mp.mpf('.5')]:
   Z=phi+(1-phi)*L;rho=phi/Z;q=rho*rho+(1-rho*rho)*Q
   value=Z*Z*q;maxvalue=max(maxvalue,value);assert value<1;count+=1
 for v0 in map(mp.mpf,['.1','1']):
  v=v0*(1+mp.j*mp.mpf('1e-4'))
  f=lambda x:abs(mp.exp(-(x+v/2)**2/(2*v))/mp.sqrt(2*mp.pi*v))
  L=mp.quad(f,[-mp.inf,0,mp.inf]);Q=mp.quad(lambda x:f(x)*kap(mp.exp(x)),[-mp.inf,0,mp.inf])/L
  value=L*L*Q;maxvalue=max(maxvalue,value);assert value<1;count+=1
print('PASS complex density contraction',count,'cells maximum Z²q',mp.nstr(maxvalue,7),flush=True)
print('PASS all additional checks',flush=True)
