"""Independent, non-certifying check of the corrected stationary points.

Uses the pre-existing two-vector moment code, independently of the scalar
parity code in gap4_certify.py. Tolerance registered before evaluation: 1e-24.
"""
import sys
from pathlib import Path
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0,str(HERE/'_deps'))
sys.path.insert(0,str(HERE.parent/'studies/memory-memA'))
import memA_checks as old
m=old.mp
m.mp.dps=80
print('Independent original two-vector recursion; five-point derivative, h=1e-7. Registered tolerance 1e-24 at corrected roots.')
for dt,st in [('.004','.599487404631930049609565775'),('.002','.599495940375631125891312954')]:
    delta,s,phi,h=map(m.mpf,[dt,st,'-.35','1e-7'])
    f=lambda z:old.series_loss(z,delta,phi,24)[0]
    deriv=(f(s-2*h)-8*f(s-h)+8*f(s+h)-f(s+2*h))/(12*h)
    print('delta',dt,'s',st,'five-point derivative',m.nstr(deriv,30))
    assert abs(deriv)<m.mpf('1e-24')
print('PASS. This independent ordinary-precision check corroborates the interval certificate; it is not the certificate.')
