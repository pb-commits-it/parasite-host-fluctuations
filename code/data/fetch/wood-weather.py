#!/usr/bin/env python3
"""Fetch registered open marine environmental data without credentials.
Successes are cached. Failures and date-command timestamps are preserved.
"""
import urllib.request,urllib.parse,json,hashlib,subprocess
from pathlib import Path
ROOT=(__import__('pathlib').Path(__file__).resolve().parent / '../..').resolve()
OUT=ROOT/'data/raw/wood/weather';OUT.mkdir(exist_ok=True)
MAN=OUT/'MANIFEST.json'
items=json.loads(MAN.read_text()) if MAN.exists() else []
def fetch(name,url,licence,version):
 p=OUT/name
 if any(r.get('file')==name and any(c in r.get('error','') for c in ['403','401']) for r in items):
  print('authorization refusal already recorded',name,flush=True);return
 if p.exists(): print('cached',name,flush=True);return
 r={'file':name,'url':url,'licence':licence,'version':version,'access_utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'OpenResearch/1.0'})
  with urllib.request.urlopen(req,timeout=60) as h: b=h.read();r['status']=h.status;r['resolved_url']=h.url
  p.write_bytes(b);r.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());print(name,len(b),flush=True)
 except Exception as e: r['error']=str(e);print(name,str(e),flush=True)
 items.append(r);MAN.write_text(json.dumps(items,indent=2)+'\n')
base='https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v6/sst.mnmean.nc'
q=urllib.parse.urlencode({'var':'sst','north':50,'south':46,'west':-128,'east':-122,'horizStride':1,'time_start':'1880-01-01T00:00:00Z','time_end':'2019-12-31T23:59:59Z','timeStride':1,'accept':'netcdf'})
fetch('ersst_v6_1880_2019.nc',base+'?'+q,'US government public-domain data; NOAA PSL attribution','ERSSTv6; live source subset 1880–2019')
fetch('ersst_v6_metadata.xml',base+'/dataset.xml','US government public-domain data','ERSSTv6')
fetch('race_rocks_monthly.html','https://racerocks.ca/archive-of-seawater-temperature-data-1921-to-present/','Publicly posted factual observations; site text copyright, retain raw only','Archive page 2019; fetched version')
fetch('dfo_station_description.html','https://www.dfo-mpo.gc.ca/science/data-donnees/lightstations-phares/index-eng.html','Government of Canada terms; documentation','page last modified 2020-05-05')
fetch('wood_doi_resolution.html','https://doi.org/10.1073/pnas.2211903120','PNAS article licence as published','DOI resolver check only')
# NCSS returned HTTP 500 on the first registered subset. The same public
# NOAA dataset has an OPeNDAP ASCII representation; no authorization bypass.
if not (OUT/'ersst_v6_1880_2019.nc').exists():
 fetch('ersst_v6_1880_2019.ascii','https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v6/sst.mnmean.nc.ascii?'+urllib.parse.quote('sst[360:1:2039][19:1:21][116:1:119]',safe=''), 'US government public-domain data; NOAA PSL attribution','ERSSTv6; same registered spatial/temporal subset')
