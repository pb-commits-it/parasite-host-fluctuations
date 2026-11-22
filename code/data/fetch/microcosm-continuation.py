#!/usr/bin/env python3
"""Fetch open primary documentation only. Never execute downloaded content."""
from pathlib import Path
import urllib.request, json, hashlib, subprocess, xml.etree.ElementTree as E
ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve(); OUT=ROOT/'data/raw/microcosm/study_docs';OUT.mkdir(exist_ok=True)
items=[('PMC7287369','ferris2020'),('PMC3768316','duncan2013'),('PMC1560244','fels2006'),('PMC5792953','dallas2018')]
log=[]
for pmc,label in items:
 u=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML';p=OUT/f'{pmc}_{label}.xml'
 try:
  if not p.exists():
   with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'microcosm-research/1.0'}),timeout=30) as r: p.write_bytes(r.read())
  rt=E.fromstring(p.read_bytes()); title=''.join(rt.find('.//article-title').itertext());body='\n'.join(''.join(q.itertext()) for q in rt.findall('.//body//p'))
  p.with_suffix('.txt').write_text(body)
  log.append({'url':u,'path':str(p.relative_to(ROOT/'data/raw/microcosm')),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'title':title,'time':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()});print(label,title)
 except Exception as e:log.append({'url':u,'error':str(e)});print(label,str(e))
(OUT/'CONTINUATION_MANIFEST.json').write_text(json.dumps(log,indent=2)+'\n')
