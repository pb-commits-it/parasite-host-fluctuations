#!/usr/bin/env python3
"""Completion checks for the registered artifacts; no new hypothesis tests."""
import sys,json,hashlib,subprocess,importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from scipy import stats
import numpy as np

def main():
 u=json.loads((a.RES/'unified_trait_summary.json').read_text());w=json.loads((a.RES/'weather_fit_summary.json').read_text());m=json.loads((a.RES/'meta_models.json').read_text());d=json.loads((a.RES/'meta_decision.json').read_text())
 required=['protocol.md','analysis.py','report.md','traits_blind.csv','figures/trait_effects.tex','figures/trait_effects.pdf','figures/host_number_decline.tex','figures/host_number_decline.pdf','results/field_records.csv','results/weather_matches.csv','results/weather_pre_analysis.json','results/meta_multistart_check.json']
 checks={'required_files':all((a.HERE/x).exists() for x in required),'thirteen_registered_unified_models':len(u)==13,'five_environment_sets':len(w)==5 and all('delta_aicc' in z or 'not_fit' in z for z in w.values()),'bootstrap_complete':d['B']==2000 and d['bootstrap_failure_count']==0,'primary_criteria_not_met':d['delta_aicc']>-2 and d['bootstrap_p']>=.05,'all_primary_individual_p_ge_005':all(z>=.05 for z in d['traits_p'])}
 fits=[(k,z) for k,z in u.items() if z.get('estimable')]
 fits += [(k+'_'+label,z[label]) for k,z in w.items() for label in ['baseline','joint'] if label in z and z[label].get('estimable',True)]
 diagnostics=[]
 for key,z in fits:
  good=z['converged'] and z['outer_grad_max']<.02 and z['inner_grad_max']<1e-5 and z['hessian_min_eigenvalue']> -1e-5
  diagnostics.append({'model':key,'pass':good,**{k:z[k] for k in ['converged','outer_grad_max','inner_grad_max','hessian_min_eigenvalue']}})
 checks['all_fit_diagnostics']=all(z['pass'] for z in diagnostics)
 chk=json.loads((a.RES/'meta_multistart_check.json').read_text());checks['meta_multistart_agrees']=all(z['improvement']<1e-5 for z in chk.values())
 checks['blind_hash']=hashlib.sha256((a.HERE/'traits_blind.csv').read_bytes()).hexdigest()=='f4d0202392697dd3cff69f10eecdcd97f6331c27386e3c82e94a5a394332fe35'
 checks['theory_hash']=hashlib.sha256((a.RES/'theory_predictions.txt').read_bytes()).hexdigest()=='c76ba5c968c6376653723e68963b739d4dd98196a99e51f05459cf4a4a6def46'
 checks['power_hash']=hashlib.sha256((a.RES/'pre_analysis_power.txt').read_bytes()).hexdigest()=='688e92311d81a3a246535e5535bcc5096fe930dcca7c64e4a1a55d23ce339f89'
 # Expanded coefficient tables include each registered sensitivity, not only
 # the selected columns printed in the short report.
 rows=[]
 for key,z in m.items():
  r=z['REML']
  if not r.get('estimable'):continue
  offset=1 if key in ['M5_traits_tax','M8_within_3plus'] else 3
  if key=='M3_nhosts_tax':continue
  for j,t in enumerate(a.TRAITS):
   i=offset+j;rows.append({'model':key,'method':'meta_REML','trait':t,'estimate':r['beta'][i],'se':r['se'][i],'ci_lo':r['ci_lo'][i],'ci_hi':r['ci_hi'][i],'p':r['pvalues'][i]})
 for key,z in u.items():
  if key=='M3_nhosts_tax' or not z.get('estimable'):continue
  offset=1 if key in ['M5_traits_tax','M8_within_3plus'] else 3
  for j,t in enumerate(a.TRAITS):
   i=offset+j;b=z['beta'][i];se=z['se'][i];rows.append({'model':key,'method':'count_GLMM_Wald','trait':t,'estimate':b,'se':se,'ci_lo':b-1.96*se,'ci_hi':b+1.96*se,'p':2*stats.norm.sf(abs(b/se))})
 a.pd.DataFrame(rows).to_csv(a.RES/'all_trait_contrasts.csv',index=False)
 versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','python-dateutil','six']}
 versions['python']=sys.version.split()[0];versions['tectonic']=subprocess.check_output(['tectonic','--version'],text=True).strip()
 a.save('software_versions',versions)
 out={'timestamp':a.stamp(),'checks':checks,'all_pass':all(checks.values()),'fit_diagnostics':diagnostics,'software':versions,'registered_model_count':len(u),'environment_set_count':len(w)}
 a.save('completion_checks',out);print(json.dumps(a.plain(out),indent=2));assert out['all_pass']

if __name__=='__main__':main()
