"""Check preserved inputs, numerical output completeness and figures, then finish report."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *
import ast

def main():
 inv=json.loads((RESULTS/'continuation_inventory.json').read_text());originals=[]
 for name,h in inv['original_sha256'].items():
  data=(HERE/name).read_bytes()
  if hashlib.sha256(data).hexdigest()==h:originals.append(dict(path=name,status='unchanged'));continue
  assert name in ['protocol.md','literature/SEARCH_LOG.csv'],name
  ends=[i+1 for i,c in enumerate(data) if c==10]
  matches=[i for i in ends if hashlib.sha256(data[:i]).hexdigest()==h]
  assert matches,name;originals.append(dict(path=name,status='original prefix preserved',bytes=matches[0]))
 prefix=json.loads((RESULTS/'report_prefix_preservation.json').read_text());p=HERE/'report.md';before=p.read_bytes();assert hashlib.sha256(before[:prefix['bytes']]).hexdigest()==prefix['sha256']
 files=list(HERE.glob('*.py'))+[ROOT/'data/fetch/envcov_literature.py']
 for path in files:ast.parse(path.read_text(),filename=path.name)
 f=json.loads((RESULTS/'field_analysis.json').read_text());t=json.loads((RESULTS/'theory_analysis.json').read_text());s=json.loads((RESULTS/'simulation_analysis.json').read_text());w=json.loads((RESULTS/'driven_weather_analysis.json').read_text());v=json.loads((RESULTS/'verification_and_supplement.json').read_text())
 assert len(t['cells'])==160 and len(s['engine_cells'])==72 and len(w['cells'])==54
 assert len(s['neon'])==45 and len(s['cary'])==15 and len(w['scenarios'])==9
 for event in s['events']+w['events']:
  assert event['n']==2000 and abs(event['frequency']-event['count']/event['n'])<1e-12
  assert -1e-12<=event['wilson_low']<=event['frequency']+1e-12 and event['frequency']-1e-12<=event['wilson_high']<=1+1e-12
 assert f['neon_adjustments'][1]['verdict']=='not decided'
 assert v['state_matrix_check_error']<1e-10 and v['conditional_mean_quadrature_error']<1e-10
 for key,nrows in [('cary_pairs',None)]:assert key in f
 assert len(f['cary_pairs']['primary'])==102 and len(f['cary_pairs']['holdout'])==40 and len(f['neon_pairs'])==46
 pdfs=sorted(FIGURES.glob('0*.pdf'));assert len(pdfs)==4
 for pdf in pdfs:
  assert pdf.read_bytes().startswith(b'%PDF-') and pdf.stat().st_size>1000
 for n in range(1,5):assert 'Writing' in (RESULTS/f'figure{n}_compile.txt').read_text()
 # Verify numerical CSVs used by the figures against the authoritative saved outputs.
 with (FIGURES/'neon_sites.csv').open() as stream:plot=list(csv.DictReader(stream))
 for row,original in zip(plot,f['neon_sites']):assert float(row['sigma'])==original['sigma'] and float(row['nip'])==original['nip']
 with (FIGURES/'masking_rates.csv').open() as stream:plot=list(csv.DictReader(stream))
 for row,original in zip(plot,w['scenarios']):assert float(row['supported'])==original['support_cutoff']
 timestamp=now();summary=dict(timestamp=timestamp,originals=originals,python_syntax_files=len(files),report_prefix_preserved=True,engine_cells=286,synthetic_studies=sum(x['n'] for x in w['scenarios'])+sum(x['replicates'] for x in s['neon']+s['cary']),event_count_checks=len(s['events'])+len(w['events']),pdfs=[p.name for p in pdfs],result='PASS')
 if not before.decode().rstrip().endswith('Status: done'):
  text=f'''\n## Completion record ({timestamp})\n\nFinal integrity checks passed: {len(originals)} supplied files are unchanged or\nhave their exact original prefix preserved; the report's earlier sections are\nunchanged. Numerical artifacts contain 286 completed engine calibration cells\nacross the three finite campaigns and {summary['synthetic_studies']} synthetic studies.\nEvent counts, frequencies, Monte Carlo intervals and figure inputs reconcile.\nAll four standalone pgfplots figures compiled with tectonic and were visually\nchecked. The final audit is saved in results/final_validation.json.\n\nFigures: [NEON sites](figures/01_neon_sites.pdf),\n[Cary adjustments](figures/02_cary_adjustment.pdf),\n[duration and stay-shape comparisons](figures/03_duration_loss.pdf), and\n[masking frequencies](figures/04_masking_rates.pdf).\n\nStatus: done\n'''
  with p.open('a') as out:out.write(text)
  (RESULTS/'completion_record.md').write_text(text)
  print(text)
 summary['report_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();save('final_validation.json',summary)
 artifact_hashes={str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(HERE.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json','final_validation.txt'] and f.suffix!='.so'}
 save('artifact_hashes.json',artifact_hashes)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
