"""Local serialization and reproducible timestamps; no raw abundance export to results."""
from pathlib import Path
import csv, gzip, json, math, subprocess
import numpy as np
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
RAW=ROOT/'data/raw/lpd'
RESULTS=HERE/'results'
def now(): return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def clean(x):
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    if path.suffix=='.gz':
        with gzip.open(tmp,'wt') as f:json.dump(clean(x),f,separators=(',',':'),allow_nan=False)
    else: tmp.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(path.read_text())
def csvwrite(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v)) if isinstance(v,(dict,list,tuple)) else clean(v) for k,v in r.items()})
