#!/usr/bin/env python3
"""Independent numerical and data-integrity checks; preserves inherited documents."""
from pathlib import Path
import sys,collections
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *

def exact_s1(x,y):
 # Enumerate count allocations of tied survival values into five amplitude cells.
 # Integer rank cross-products and combinatorial weights avoid Monte Carlo error.
 rx=np.rint(2*stats.rankdata(x)).astype(int);ry=np.rint(2*stats.rankdata(y)).astype(int)
 levels=sorted(set(rx));cap=[sum(rx==v) for v in levels];cats=sorted(collections.Counter(ry).items(),key=lambda z:z[1]);base=ry.max();cats=[(r,c) for r,c in cats if r!=base];target=int(rx@(ry-base));weights=collections.defaultdict(int)
 def allocations(n,caps):
  if len(caps)==1:
   if n<=caps[0]:yield (n,)
   return
  for a in range(min(n,caps[0])+1):
   for rest in allocations(n-a,caps[1:]):yield (a,)+rest
 def rec(j,caps,score,weight):
  if j==len(cats):weights[score]+=weight;return
  rank,count=cats[j]
  for take in allocations(count,caps):
   w=math.prod(math.comb(c,t) for c,t in zip(caps,take));rec(j+1,tuple(c-t for c,t in zip(caps,take)),score+(rank-base)*sum(a*b for a,b in zip(levels,take)),weight*w)
 rec(0,tuple(cap),0,1);total=sum(weights.values());expect=math.factorial(len(y))//math.prod(math.factorial(c) for c in collections.Counter(ry).values());assert total==expect
 return dict(p_less_exact=sum(v for k,v in weights.items() if k<=target)/total,p_greater_exact=sum(v for k,v in weights.items() if k>=target)/total,weighted_allocations=total,distinct_rank_statistics=len(weights))

def main():
 SUMMARY.update(json.loads((OUT/'summary_metrics.json').read_text()));checks=[]
 audit=json.loads((OUT/'continuation_audit.json').read_text())
 for name,meta in audit['originals'].items():
  b=(HERE/name).read_bytes()[:meta['bytes']];ok=hashlib.sha256(b).hexdigest()==meta['sha256'];assert ok;checks.append(dict(check='inherited prefix',file=name,pass_=ok))
 manifest=json.loads((RAW/'MANIFEST.json').read_text());verified=0;missing=[];not_downloaded=collections.Counter()
 for r in manifest:
  if not r.get('downloaded'):not_downloaded[r.get('reason','unknown')]+=1;continue
  if r.get('sha256'):
   p=RAW/r['path']
   if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p);verified+=1
   else:missing.append(r['path'])
 record('provenance_audit',dict(original_hashes_verified=verified,missing_manifest_files=missing,not_downloaded_manifest_entries=sum(not_downloaded.values()),not_downloaded_reasons=dict(not_downloaded),refused_manifest_entries=sum(n for reason,n in not_downloaded.items() if 'refused' in reason.lower()),git_ignore_check='unavailable: supplied directory has no .git repository'))
 u=dh.read_csv_dicts(str(OUT/'s1_units.csv'));x=np.array([num(r['amplitude']) for r in u]);y=np.array([num(r['survival_transfer']) for r in u]);record('T1a_exact',exact_s1(x,y))
 record('T1a_by_D',dh.perm_spearman(np.array([num(r['D']) for r in u]),y,100000,SEED))
 # Bootstrap uncertainty for rho is descriptive, not the registered test.
 rng=np.random.default_rng(SEED);boot=[]
 for _ in range(10000):
  ids=rng.integers(0,len(x),len(x));boot.append(stats.spearmanr(x[ids],y[ids]).statistic)
 record('T1a_interval',dict(lo=np.nanquantile(boot,.025),hi=np.nanquantile(boot,.975),bootstrap_units=30))
 # Endpoint statistics re-derived directly from raw arrays with no helper growth code.
 p=xlsx('duncan2013_paramecium_holospora','Duncan_Parasite_time_07_13.xlsx',1);dd={}
 for key,rr in groups(p,['Metapopulation']).items():
  if rr[0]['Temperature']!='Variable':continue
  by={int(r['Time']):r for r in rr};rate=math.log(((num(by[16]['Number infected'])+.5)/(num(by[16]['Total'])+1))/((num(by[0]['Number infected'])+.5)/(num(by[0]['Total'])+1)))/16;dd[key[0]]=(rr[0]['Red/White Variation'],rate)
 delta=np.mean([r[1] for r in dd.values() if r[0]=='Red'])-np.mean([r[1] for r in dd.values() if r[0]=='White']);assert abs(delta-SUMMARY['T2a']['diff'])<1e-12;checks.append(dict(check='T2a raw ratio reconstruction',value=delta,tolerance=1e-12,pass_=True))
 # Independent score-equation solution for spore decay constant.
 rr=[r for r in read('dallas2016_metschnikowia_tempvar','sporeSurv.csv') if r['trt']=='0'];t=np.array([num(r['day']) for r in rr]);dead=np.array([num(r['dead']) for r in rr]);n=np.array([num(r['total']) for r in rr]);mu=optimize.brentq(lambda m:np.sum(t*dead/np.expm1(m*t)-t*(n-dead)),.001,.5);assert abs(mu-SUMMARY['S4']['nominal_mu'])<1e-7;checks.append(dict(check='spore exponential score equation',value=mu,tolerance=1e-7,pass_=True))
 # Stable scaled propagator vs old reference on safe arguments (including Erlang).
 for ko,ki in [(1,1),(4,4),(1,16)]:
  for d in [1,.5,.1,0]:
   a=floquet(2,7,k_out=ko,k_in=ki,d_wait=d,period=9);b=dh.floquet_loss_D(2,7,ko,ki,d,9);assert abs(a-b)<1e-9;checks.append(dict(check='scaled versus unscaled Floquet',k_out=ko,k_in=ki,d=d,error=abs(a-b),tolerance=1e-9,pass_=True))
 # Field dates are independently reconstructed from published Julian columns.
 rr=read('davenport2024a_spores_outbreaks','Clean_Data_2021_Lakes_All_Host_Densities.csv');import datetime
 corrected=0
 for r in rr:
  dt=datetime.date(int(r['Year']),1,1)+datetime.timedelta(days=int(r['Julian'])-1)
  if r['Date']=='NA':corrected+=1
  else:assert dt==datetime.datetime.strptime(r['Date'],'%m/%d/%y').date()
 checks.append(dict(check='field missing Date reconstructed from Year/Julian',value=corrected,pass_=True))
 # All T2 MC tails agree with exact labels; exact p is available, not a claim of precision from 10k.
 for name,r in SUMMARY.items():
  if name.startswith('T2') and isinstance(r,dict) and 'p_less_exact' in r:
   assert (r['p_less']<.05)==(r['p_less_exact']<.05);assert (r['p_greater']<.05)==(r['p_greater_exact']<.05)
 save('independent_audit',checks);record('audit_completed',dict(time=now(),checks=len(checks),passed=True))
if __name__=='__main__':main()
