"""Export standalone pgfplots figures and their exact CSV plotting data."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import *

def csvout(name,rows):
 with (FIGURES/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tex(name,body):
 (FIGURES/name).write_text(r'''\documentclass[tikz,border=6pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\begin{document}
\begin{tikzpicture}
'''+body+'\n\\end{tikzpicture}\n\\end{document}\n')
def main():
 d=json.loads((RESULTS/'field_analysis.json').read_text());t=json.loads((RESULTS/'theory_analysis.json').read_text());s=json.loads((RESULTS/'driven_weather_analysis.json').read_text())
 csvout('neon_sites.csv',[dict(site=r['site'],sigma=r['sigma'],nip=r['nip'],climate=r['PC1_env']) for r in d['neon_sites']])
 tex('01_neon_sites.tex',r'''\begin{axis}[width=12cm,height=8cm,xlabel={Mouse log-abundance SD},ylabel={Nymphal infection prevalence},xmin=.32,xmax=1.10,ymin=0,ymax=.30,grid=major,title={Six NEON sites: climate adjustment remains inconclusive}]
\addplot[only marks,scatter,scatter src=explicit,mark=*,mark size=3pt,nodes near coords,point meta=explicit symbolic,every node near coord/.append style={font=\small,anchor=south}] table[x=sigma,y=nip,meta=site,col sep=comma]{neon_sites.csv};
\end{axis}''')
 labels=['Unadjusted','Nymph PC1','Larval PC1','Six covariates'];ks=['unadjusted','PC1_nymph','PC1_larva','all_six'];rows=[]
 for per in ['primary','holdout']:
  for i,k in enumerate(ks):
   r=next(x for x in d['cary_slopes'] if x['period']==per and x['covariate']==k);rows.append(dict(period=per,index=i,b=r['b'],error=2*r['se']))
 for per in ['primary','holdout']:csvout('cary_'+per+'.csv',[{k:v for k,v in r.items() if k!='period'} for r in rows if r['period']==per])
 tex('02_cary_adjustment.tex',r'''\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.6cm},width=8cm,height=8cm,xtick={0,1,2,3},xticklabels={Unadjusted,Nymph PC1,Larval PC1,Six covariates},x tick label style={rotate=30,anchor=east},ylabel={Mouse slope (bars: two bootstrap SE)},grid=major]
\nextgroupplot[title={Primary period}]
\addplot[only marks,mark=*,error bars/.cd,y dir=both,y explicit] table[x=index,y=b,y error=error,col sep=comma]{cary_primary.csv};
\addplot[dashed,domain=-.4:3.4]{0};
\nextgroupplot[title={Holdout; all-six estimate unstable}]
\addplot[only marks,mark=*,error bars/.cd,y dir=both,y explicit] table[x=index,y=b,y error=error,col sep=comma]{cary_holdout.csv};
\addplot[dashed,domain=-.4:3.4]{0};
\end{groupplot}''')
 csvout('duration_loss.csv',[dict(index=i,site=r['site'],borrowed=r['old_D'],thermal=r['thermal_same_clock_D'],memoryless=r['fine_memoryless_D'],chain=r['fine_chain_D']) for i,r in enumerate(t['shifts'])])
 tex('03_duration_loss.tex',r'''\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.5cm},width=8cm,height=7cm,xtick={0,1,2,3,4,5,6},xticklabels={Cary,BLAN,HARV,ORNL,SCBI,SERC,TREE},x tick label style={rotate=45,anchor=east},ylabel={Loss of log reproduction number},grid=major,legend style={font=\small,at={(.5,-.37)},anchor=north}]
\nextgroupplot[title={Original clock; independent host steps}]
\addplot[mark=square*] table[x=index,y=borrowed,col sep=comma]{duration_loss.csv};\addlegendentry{Borrowed tick duration}
\addplot[mark=*] table[x=index,y=thermal,col sep=comma]{duration_loss.csv};\addlegendentry{Thermal duration}
\nextgroupplot[title={Six-day clock; assumed positive memory}]
\addplot[mark=square*] table[x=index,y=memoryless,col sep=comma]{duration_loss.csv};\addlegendentry{Memoryless tick stay}
\addplot[mark=*] table[x=index,y=chain,col sep=comma]{duration_loss.csv};\addlegendentry{Chain matched to model spread}
\end{groupplot}''')
 rows=[]
 for i,r in enumerate(s['scenarios']):rows.append(dict(index=i,gamma=r['gamma'],kappa=r['kappa'],negative=r['negative_unadjusted'],supported=r['support_cutoff'],adjusted=r['support_adjusted']))
 csvout('masking_rates.csv',rows)
 tex('04_masking_rates.tex',r'''\begin{axis}[width=15cm,height=8cm,ymin=0,ymax=1,xtick={0,1,2,3,4,5,6,7,8},xticklabels={$0$,$.5$,$1$,$0$,$.5$,$1$,$0$,$.5$,$1$},xlabel={Host elasticity within weather sensitivities $-0.2$, $0$, $+0.2$},ylabel={Frequency in 2,000 synthetic studies},grid=major,legend pos=north west,title={Negative signs are common; registered support remains uncommon}]
\addplot[mark=*] table[x=index,y=negative,col sep=comma]{masking_rates.csv};\addlegendentry{Negative unadjusted sign}
\addplot[mark=square*] table[x=index,y=supported,col sep=comma]{masking_rates.csv};\addlegendentry{Registered unadjusted support}
\addplot[mark=triangle*] table[x=index,y=adjusted,col sep=comma]{masking_rates.csv};\addlegendentry{Climate-adjusted support}
\end{axis}''')
 (FIGURES/'PLOTTING.md').write_text('''# Plotting notes

Each numbered .tex is a standalone pgfplots source with companion CSVs. Run
`tectonic 01_neon_sites.tex` (and the other files) from this directory.
No matplotlib is required. CSVs preserve full numerical precision.

1. Site scatter: point labels are site codes. Climate-adjusted inference is in
   results/neon_adjustments.csv; the scatter is the unadjusted data.
2. Cary slopes: error bars show **two bootstrap standard errors**, not percentile
   confidence limits. The all-six-weather holdout estimate is an instability
   diagnostic and should not be read as a reliable adjusted causal effect.
3. Left: change duration on the same original clock. Right: compare stay shapes
   on the same six-day clock with the same assumed positive host memory. Do not
   interpret the difference between panels as a measured effect of temperature.
4. Coupled survival/development engine scenarios: two thousand studies per cell.
   Monte Carlo intervals and exact event counts are in the results CSVs. The
   survival response is illustrative; these are conditional frequencies.
''')
 print('FIGURES: four standalone pgfplots sources and five CSV tables',flush=True)
if __name__=='__main__':main()
