#!/usr/bin/env python3
"""Verify/reuse registered open weather downloads; fetch only missing files.
All requests use a generic User-Agent; no credential or account is used.
"""
from pathlib import Path
import json, hashlib, subprocess, urllib.request
ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve();RAW=ROOT/'data/raw/microcosm'
manifest=json.loads((RAW/'MANIFEST.json').read_text());out=[]
for item in manifest:
 if not item['path'].startswith('weather/'):continue
 p=RAW/item['path'];rec=dict(item)
 if not p.exists():
  try:
   with urllib.request.urlopen(urllib.request.Request(item['url'],headers={'User-Agent':'microcosm-research/1.0'}),timeout=30) as r:p.write_bytes(r.read())
   rec['action']='downloaded_missing'
  except Exception as e:rec['error']=str(e)
 else:rec['action']='reused_existing'
 if p.exists():
  rec['verified_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();rec['matches_original_hash']=rec['verified_sha256']==item.get('sha256');assert rec['matches_original_hash'],str(p)
 rec['verified_utc']=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip();out.append(rec)
(RAW/'weather/MANIFEST_verified.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([{'path':x['path'],'action':x['action'],'matches_original_hash':x.get('matches_original_hash')} for x in out],indent=2))
