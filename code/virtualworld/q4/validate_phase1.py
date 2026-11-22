"""Reproduce registered phase-1 cell before Q4: tolerance fixed in protocol.md."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0,str(P.parent))
import json, subprocess, time, hashlib
import numpy as np
from world.model import ring
from world.hosts import lognormal
from world.invasion import products, ring_coefficient, resolvent_coefficient, se

if __name__=='__main__':
    dest=P/'phase1-validation.json'
    if dest.exists(): raise RuntimeError('append-only output exists')
    start=time.monotonic()
    m=ring([2,5]); K=ring_coefficient([2,5],.6)
    h=lognormal(65536+2048,32,.03,.6,151120,antithetic=True)
    r,_,_=products(m,h,0.,2048,32,control=1/7,control_mean=1.)
    values=-7*r/.03**2
    mean=float(values.mean()); error=abs(mean-K); stderr=se(values)
    tolerance=.05*K+4*stderr
    algebra=abs(K-resolvent_coefficient(m,.6))
    out=dict(date=subprocess.check_output(['date','+%Y-%m-%d %H:%M:%S %Z'],text=True).strip(),
        K=K,estimate=mean,se=stderr,error=error,tolerance=tolerance,
        precision_limit=.03*K,resolvent_error=algebra,seed=151120,
        steps=65536,burn=2048,pairs=32,seconds=time.monotonic()-start,
        protocol_sha256=hashlib.sha256((P/'protocol.md').read_bytes()).hexdigest(),
        passed=bool(error<=tolerance and stderr<=.03*K and algebra<=1e-10))
    dest.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
    if not out['passed']: sys.exit(1)
