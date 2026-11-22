#!/usr/bin/env python3
"""Remaining registered descriptive extractions and source-derived parameter ledger."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *

def main():
 SUMMARY.update(json.loads((OUT/'summary_metrics.json').read_text()))
 # S1 growth assays: each column is one clone x resource; slopes on the original time grid.
 rr=dh.read_xlsx_sheet(str(RAW/'ferris2020_phage_oscillations/rspb20200787_si_002.xlsx'),2);out=[];time=np.array([num(r[0]) for r in rr[2:]])/3600
 for j in list(range(2,38))+list(range(39,75)):
  v=np.array([num(r[j]) if len(r)>j else np.nan for r in rr[2:]]);good=np.isfinite(time+v);t=time[good];v=v[good]
  out.append(dict(strain='ancestor' if j<38 else 'resistant',clone=rr[0][j],KB=num(rr[1][j]),max_OD=np.max(v),max_adjacent_OD_per_hour=np.max(np.diff(v)/np.diff(t)),measure='raw adjacent OD slope; not adsorption, transmission, or sustained exponential growth'))
 save('s1_growth_assay',out)
 # At each treatment, count phage fade-outs; demographic transfer bottleneck is descriptive.
 u=dh.read_csv_dicts(str(OUT/'s1_units.csv'));pred=dh.read_csv_dicts(str(OUT/'s1_floquet.csv'));sens=[]
 for d in [1,.5,.1,.03]:
  for k in [1,16]:
   lut={r['treatment']:num(r['D']) for r in pred if num(r['d'])==d and int(r['k_in'])==k};res=slope([lut[r['treatment']] for r in u],[num(r['early_logPFU']) for r in u]);sens.append(dict(d=d,k_in=k,**res))
 save('s1_sensitivity_tests',sens)
 # Both host control means and the realized variability are retained, not assumed matched.
 h=dh.read_csv_dicts(str(OUT/'s2_host_units.csv'));hs=[]
 for infection in ['Uninfected','Infected']:
  for noise in ['Red','White']:
   r=[r for r in h if r['infection']==infection and r['noise']==noise and r['temperature']=='Variable']
   for metric in ['mean_H','sd_logH','sd_logH_detrended','r1','r2']:hs.append(dict(infection=infection,noise=noise,metric=metric,**mean_ci([num(q[metric]) for q in r])))
 save('s2_noise_host_summary',hs)
 # Native units are preserved; spore concentration is per litre but the host-density
 # unit is not explicit in the deposited README, so no spores-per-host conversion.
 spec=[('Metsch_Mean_Spores_Outbreak.csv','Metschnikowia','Mean'),('Pasteuria_Mean_Spores_and_Percent_Infected_Outbreak.csv','Pasteuria','Mean'),('Brood_Mean_Spores_and_Percent_Infected_OutbreakSize.csv','Blastulidium','Mean'),('MicG_Mean_Spores_and_Percent_Infected_OutbreakSize.csv','Ordospora pajunii','Mean'),('Spiro_Mean_Spores_and_Percent_Infected.csv','Spirobacillus','Mean')];spores=[]
 for file,parasite,col in spec:
  for (lake,),rr in groups(read('davenport2024a_spores_outbreaks',file),['Lake']).items():
   vals=[num(r[col]) for r in rr];spores.append(dict(lake=lake,parasite=parasite,unit='reported spore equivalents per L',**mean_ci(vals),minimum=np.nanmin(vals),maximum=np.nanmax(vals)))
 save('s6_spore_concentrations',spores)
 params=[dict(system='S1',parameter='tau_out',value=.35,unit='day',basis='illustrative inherited assumption; no adsorption calibration located'),dict(system='S1',parameter='tau_in',value=.03,unit='day',basis='illustrative inherited assumption; no lysis assay located'),dict(system='S2',parameter='tau_out',value=2,unit='day',basis='illustrative; free infective-form survival is not measured here'),dict(system='S2',parameter='tau_in',value=7,unit='day',basis='approximate latency from Duncan 2013; not entire infected lifespan'),dict(system='S3',parameter='filtering_gamma',value=.00453,unit='per model host per day; source model units',basis='Dallas2017.Rmd parameter listing; literature input, not fitted here'),dict(system='S3',parameter='per_spore_infectivity_u',value=.003,unit='probability per contact',basis='Dallas2017.Rmd parameter listing; illustrative/literature model input'),dict(system='S3',parameter='gamma_times_u',value=.00453*.003,unit='source model composite transmission coefficient',basis='product of source constants; not estimated from outbreak outcomes'),dict(system='S3',parameter='spore_death',value=.75,unit='per day',basis='Dallas2017.Rmd model input; differs from S4 dye assay'),dict(system='S3',parameter='spore_yield',value=10000,unit='per infected host',basis='Dallas2017.Rmd model input; not measured in expData.csv')]
 save('parameter_provenance',params)
 record('supplementary_completed',dict(time=now(),growth_columns=len(out),spore_lake_series=len(spores),S1_sensitivity_labels=sorted(set(r['status'] for r in sens))))
if __name__=='__main__':main()
