"""Fetch public CC0 Zenodo archives anonymously; save provenance and checksums."""
from pathlib import Path
import urllib.request, json, hashlib, subprocess, time
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ecology-research/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        return r.read(),r.geturl(),r.status

if __name__=='__main__':
    if (P/'fetch-manifest.json').exists(): raise RuntimeError('append-only output exists')
    out=[]; start=time.monotonic()
    for record in ('7972066','4976742'):
        d=P/'raw'/record; d.mkdir(parents=True,exist_ok=True)
        url=f'https://zenodo.org/api/records/{record}'
        try:
            data,final,status=fetch(url); (d/'metadata.json').write_bytes(data)
            meta=json.loads(data)
            out.append(dict(url=url,final_url=final,status=status,sha256=hashlib.sha256(data).hexdigest(),
                licence=meta.get('metadata',{}).get('license'),record=record))
            files=[(f['key'],f['links']['self'],f.get('checksum')) for f in meta['files']]
        except Exception as e:
            out.append(dict(url=url,error=str(e),record=record))
            names=(['README.md','BeetleTeamTriboliumDataArchive.xlsx'] if record=='7972066' else
                ['README_for_Desharnais_et_al_Tribolium_synchrony_data.txt','Desharnais_et_al_Tribolium_synchrony_data.csv'])
            files=[(name,f'https://zenodo.org/records/{record}/files/{name}?download=1',None) for name in names]
        for name,url,checksum in files:
            if time.monotonic()-start>300: raise RuntimeError('fetch budget')
            try:
                data,final,status=fetch(url); (d/name).write_bytes(data)
                if checksum and checksum.startswith('md5:'):
                    assert hashlib.md5(data).hexdigest()==checksum[4:]
                out.append(dict(record=record,name=name,url=url,final_url=final,status=status,
                    bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),archive_checksum=checksum))
                print(record,name,status,len(data),flush=True)
            except Exception as e:
                out.append(dict(record=record,name=name,url=url,error=str(e)))
                print(record,name,'ERROR',str(e),flush=True)
    (P/'fetch-manifest.json').write_text(json.dumps(dict(date=subprocess.check_output(['date','+%Y-%m-%d %H:%M:%S %Z'],text=True).strip(),
        user_agent='ecology-research/1.0',seconds=time.monotonic()-start,requests=out),indent=2)+'\n')
