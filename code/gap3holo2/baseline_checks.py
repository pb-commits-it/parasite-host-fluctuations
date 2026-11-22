"""Known deterministic threshold and finite matrix slope core.
Tolerance fixed in plan.md for deterministic root: 1e-35.
Finite matrix comparisons use exact rational arithmetic (zero tolerance).
"""
from fractions import Fraction as Q
import json
from pathlib import Path
import mpmath as mp
mp.mp.dps=65
for sv in ['.3','.6','.9']:
 s=mp.mpf(sv);a=1-s
 root=(s+mp.sqrt(s*s+4*a))/2
 assert abs(mp.log(root))<mp.mpf('1e-35')
print('PASS deterministic D=0 at all 3 s',flush=True)
count=0
for s in [Q(3,10),Q(6,10),Q(9,10)]:
 for c in [Q(1,10),Q(1,2),Q(2)]:
  for h0 in [Q(1,10),Q(1),Q(10)]:
   for h1 in [Q(1,10),Q(1),Q(10)]:
    for h2 in [Q(1,10),Q(1),Q(10)]:
     vals=[s**3+s*c*(h2+h1),c*h0*(s*s+c*h2),s*s+c*h1,s*c*h0]
     ders=[s*c*(h2+h1),c*h0*(s*s+2*c*h2),c*h1,s*c*h0]
     beta=c*min(h1,h2)/(s*s+c*min(h1,h2))
     assert all(d>=beta*v for d,v in zip(ders,vals));count+=1
print('PASS three-step elasticity',count,'exact rational cases',flush=True)
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
results=json.loads((P/'analytic_results.json').read_text())
for sv in ['0.3','0.6','0.9']:
 d=mp.mpf(results[sv]['threshold_p200']);ell=mp.log(200)-mp.digamma(200)
 assert 0<d<ell<mp.mpf(1)/199<mp.mpf('.1')
print('PASS gamma root bracket at p=200 in all 3 s',flush=True)
