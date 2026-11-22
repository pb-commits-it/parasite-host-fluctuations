"""Verify key full-text model sources from the public Europe PMC archive. No credentials."""
from pathlib import Path
import urllib.request,json,hashlib,subprocess,time
out=(__import__('pathlib').Path(__file__).resolve().parent / '../raw/envcov/literature').resolve();out.mkdir(exist_ok=True)
rows=[]
for pmc in ['PMC6766261','PMC12040446']:
 url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML'
 p=out/(pmc+'.xml')
 try:
  if not p.exists():
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'envcov-study-fetch/1.0 (python-urllib)'}),timeout=25) as r:body=r.read()
   p.write_bytes(body)
  body=p.read_bytes();rows.append(dict(path=p.name,url=url,sha256=hashlib.sha256(body).hexdigest(),bytes=len(body),accessed_utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),licence='See article permissions element',version='Europe PMC full text XML'))
  print(p.name,len(body))
 except Exception as e:print(pmc,type(e).__name__,str(e))
 time.sleep(.4)
(out/'MANIFEST.json').write_text(json.dumps(rows,indent=2)+'\n')
