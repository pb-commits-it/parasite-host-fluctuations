#!/usr/bin/env python3
"""Audit provenance/locked traits and preserve source-read limitations.
This does not recode traits or alter a registered criterion after outcomes.
"""
import sys,json,csv,hashlib,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a

def main():
 p=a.HERE;df,tr=a.load();manifest=json.loads((a.RAW/'MANIFEST.json').read_text());rows=[]
 for item in manifest['items']:
  rel=item.get('path')
  if rel:
   fp=a.RAW/rel
   if fp.exists():
    b=fp.read_bytes();sha=hashlib.sha256(b).hexdigest();md5=hashlib.md5(b).hexdigest()
    rows.append({'file':rel,'bytes':len(b),'sha256':sha,'recorded_sha256_matches':sha==item['sha256'] if 'sha256' in item else 'not recorded','recorded_md5_matches':md5==item['md5'] if 'md5' in item else 'not recorded'})
 a.pd.DataFrame(rows).to_csv(a.RES/'raw_integrity_audit.csv',index=False)
 # One paper/monograph card per inherited author-year reference; read status is
 # explicitly unknown/unread, never inferred from its appearance in the ledger.
 refs=['Schmidt 1985','Nickol et al. 2002','Valtonen et al. 2001','Xylander 2005','Williams et al. 1987','Mudry & Dailey 1971','Kuchta et al. 2008','Palm 2004','Jensen & Bullard 2010','Chervy 2002','Kabata 1979','Kabata 1988','Boxshall & Halsey 2004','Burreson 1975','Sawyer 1986','Bakke et al. 2007','Kearn 1998','Whittington & Chisholm 2008','Anderson 2000','Mattiucci & Nascetti 2008','Moravec et al. 1981','Moravec 2001','Koie 2000','Margolis & Moravec 1987','Sakaguchi & Fukuhara 1975','Koie 1993','Moravec 2004','Smith 1967','Koie 1982','Cribb et al. 2003','Koie 1979','Koie 1989','Koie 1990','Koie 1995','Gibson & Bray 1979','Margolis & Boyce 1969','Bray, Gibson & Jones 2008','Koie 1985','Koie 1975','Koie 1981','Hunninen & Cable 1943','Olson 1978']
 cards=p/'literature_cards';cards.mkdir(exist_ok=True)
 for ref in refs:
  slug=re.sub('[^a-z0-9]+','-',ref.lower()).strip('-')
  fp=cards/(slug+'.md')
  if not fp.exists():fp.write_text(f'# {ref}\n\nAudit date: {a.stamp()}\n\nSource: inherited build_traits_blind.py and trait_sources.md. Full bibliographic identity was not supplied in the ledger.\n\nRead status: neither full text nor abstract independently read in this continuation; earlier read status undocumented. DOI: not resolved or verified.\n\nRole: background for inherited life-cycle codes only. The original blind codes are retained. This citation cannot clear or sink a novelty claim, and does not by itself verify measured host variance, stage duration, or residence-time spread.\n')
 (cards/'wood-2023.md').write_text(f'# Wood et al. 2023\n\nAudit date: {a.stamp()}\n\nA reconstruction of parasite burden reveals one century of climate-associated parasite decline. DOI: 10.1073/pnas.2211903120. Local open full text: data/raw/wood/wood_epmc/PMC9934024.xml. Results, discussion and relevant Methods were read; no claim of a cover-to-cover SI read. DOI resolver returned HTTP 403; DOI independently confirmed in XML, not successfully resolved by this continuation.\n\nThe original paper already examines temperature, pollutants and fish density. Its host-number result is the reproduction target; its environment analysis is prior art, not novelty supplied by this reanalysis.\n')
 for slug,doi,name in [('benesh-2017','10.1002/ecy.1680','Helminth life-cycle database'),('benesh-2021-data','10.5061/dryad.gtht76hmm','Facultative hosts / life history data'),('stephens-2017','10.1002/ecy.1799','Global Mammal Parasite Database')]:
  (cards/(slug+'.md')).write_text(f'# {name}\n\nAudit date: {a.stamp()}\n\nDOI: {doi}. Downloaded data/readmes inspected, publication full text and abstract not independently read in this continuation. DOI resolver not independently checked; inherited manifest contains repository metadata. Codes are retained as inherited; no novelty judgment is made.\n')
 tr[['psite_code','par_type','table_s3_id','stage_in_fish','resolved_taxon','sources']].to_csv(a.RES/'trait_source_audit.csv')
 summary={'timestamp':a.stamp(),'manifest_files_audited':len(rows),'hash_mismatches':[r['file'] for r in rows if r['recorded_sha256_matches'] is False or r['recorded_md5_matches'] is False],'trait_rows':len(tr),'missing_source_strings':int(tr.sources.isna().sum()),'source_cards':len(list(cards.glob('*.md'))),'trait_codes_revised':0,'git_ignore_reverified':False,'git_reason':'supplied copy is not a Git repository','blind_hashes_match':all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in {'traits_blind.csv':'f4d0202392697dd3cff69f10eecdcd97f6331c27386e3c82e94a5a394332fe35','results/theory_predictions.txt':'c76ba5c968c6376653723e68963b739d4dd98196a99e51f05459cf4a4a6def46','results/pre_analysis_power.txt':'688e92311d81a3a246535e5535bcc5096fe930dcca7c64e4a1a55d23ce339f89'}.items())}
 a.save('artifact_audit',summary);print(json.dumps(summary,indent=2));assert not summary['hash_mismatches'] and summary['blind_hashes_match']

if __name__=='__main__':main()
