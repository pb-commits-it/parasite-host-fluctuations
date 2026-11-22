"""Registered iid extension to order 60, 180/260 digits; tolerance 1e-35."""
import importlib.util,json,time
from pathlib import Path
from decimal import Decimal as D
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec=importlib.util.spec_from_file_location('own',P/'checks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=json.loads((P/'coefficients_K32_P100.json').read_text());b=json.loads((P/'coefficients_K32_P160.json').read_text())
worst=D(0)
for key,x in b.items():
 err=m.compare(a[key]['coefficients'],x['coefficients']);worst=max(err,worst)
 assert err<D('1e-35'),(key,err)
print('PASS order32 all 384 coefficients, 100 vs 160 digits',worst,flush=True)
start=time.time();out={}
for s in ['0.3','0.6','0.9']:
 for host in ['gamma','lognormal']:
  fn=lambda p:m.gamma_stein(s,60,p) if host=='gamma' else m.iid_moments(s,host,60,p)
  lo=fn(180);hi=fn(260);err=m.compare(lo,hi)
  assert err<D('1e-35')
  key=f'{host}/s={s}/phi=0';out[key]={'K':60,'precision':260,'coefficients':[str(v) for v in hi]}
  print('PASS order60',key,'precision error',err,'last',f'{hi[-1]:.12E}','seconds',round(time.time()-start,1),flush=True)
  (P/'coefficients_iid_K60.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS high order extension',flush=True)
