#!/usr/bin/env python3
"""Standalone pgfplots figures from archived results; no matplotlib needed."""
from pathlib import Path
import csv,sys,json
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();R=P/'results';F=P/'figures';F.mkdir(exist_ok=True)
def read(n):return list(csv.DictReader((R/n).open()))
def data(name,rows):
 with (F/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tex(name,xlabel,ylabel,title,plots,opts=''):
 doc=r'''\documentclass[tikz,border=6pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[width=13cm,height=8cm,grid=major,grid style={gray!20},
font=\small,legend style={font=\footnotesize},legend pos=outer north east,
xlabel={'''+xlabel+'},ylabel={'+ylabel+'},title={'+title+'},'+opts+']\n'+plots+r'''
\end{axis}
\end{tikzpicture}
\end{document}
'''
 (F/name).write_text(doc)
rows=read('s1_floquet.csv');plots=''
for d,c in [('1','black'),('0.5','blue'),('0.1','orange'),('0.03','teal')]:
 sub=sorted([r for r in rows if r['d']==d and r['k_in']=='1'],key=lambda r:float(r['amplitude']));n='phage_d'+d.replace('.','')+'.csv';data(n,sub)
 plots+=r'\addplot['+c+r',thick,mark=*] table[x=amplitude,y=D,col sep=comma]{'+n+'};\n'+r'\addlegendentry{$d='+d+'$}\n'
tex('fig1_phage.tex','Nominal resource amplitude (inferred code mapping)','$D$ (conditional threshold loss)','Phage: reconstructed square-wave host schedule',plots)
rows=read('s2_trajectories.csv');plots=''
for t,n,c,label in [('Variable','White','blue','White'),('Variable','Red','red','Red'),('23','NA','black','Constant 23 C'),('5','NA','teal','Constant 5 C')]:
 sub=[r for r in rows if r['temperature']==t and (t!='Variable' or r['noise']==n)]
 if not sub:continue
 sub=sorted(sub,key=lambda r:int(r['day']));fn='duncan_'+label.replace(' ','_')+'.csv';data(fn,sub)
 plots+=r'\addplot['+c+r',thick,mark=*] table[x=day,y=mean,col sep=comma]{'+fn+'};\n'+r'\addlegendentry{'+label+'}\n'
tex('fig2_duncan.tex','Day','Mean prevalence','Holospora: early and late contrasts',plots,'xmin=0,xmax=56,ymin=0,ymax=.65,')
rows=sorted(read('s3_threshold.csv'),key=lambda r:float(r['H0_L']));data('daphnia_threshold.csv',rows)
plots=''
for which,c in [('primary','blue'),('secondary','orange')]:plots+=r'\addplot['+c+r',thick,mark=*] table[x=H0_L,y='+which+r'_fraction,col sep=comma]{daphnia_threshold.csv};'+'\n'+r'\addlegendentry{'+which+' infection}\n'
tex('fig3a_threshold.tex','Initial hosts per litre','Fraction of populations infected','Metschnikowia: initial-density manipulation',plots,'xmode=log,log basis x=2,ymin=0,ymax=1,xtick={20,40,80,160,320,640},xticklabels={20,40,80,160,320,640},')
rows=read('s34_prediction_grid.csv');plots=''
for k,d,c in [('1','waste','blue'),('1','waiting','teal'),('16','waste','red'),('16','waiting','orange')]:
 sub=sorted([r for r in rows if r['k_in']==k and r['reading']==d],key=lambda r:float(r['scale_days']));fn='prediction_'+k+'_'+d+'.csv';data(fn,sub)
 for y,sty,label in [('D_Floquet','dashed','periodic'),('D_telegraph','solid','telegraph')]:plots+=r'\addplot['+c+','+sty+r',thick,mark=*] table[x=scale_days,y='+y+',col sep=comma]{'+fn+'};\n'+r'\addlegendentry{$k='+k+'$, '+d+', '+label+'}\n'
tex('fig3b_predictions.tex','Periodic cycle or random mean dwell (days)','$D$ (conditional threshold loss)','Cross-species Daphnia calibration scenario',plots)
rows=read('weather_daily_14d.csv');plots=''
for lake,c in [('Bishop','blue'),('Cedar','red'),('CrookedW','orange'),('LilAp','teal'),('Mill','violet'),('Walsh','black')]:
 sub=[r for r in rows if r['lake']==lake];fn='weather_'+lake+'.csv';data(fn,sub)
 plots+=r'\addplot['+c+r',thick] table[x=day_of_year,y=tmean,col sep=comma]{'+fn+'};\n'+r'\addlegendentry{'+lake+'}\n'
tex('fig4_weather.tex','Day of year, 2021','Preceding 14-day mean air temperature (C)','Daymet at inherited approximate lake coordinates',plots,'xmin=150,xmax=320,')
(F/'PLOTTING.md').write_text('''# Plotting notes

The `.tex` files are standalone pgfplots documents. Compile from this directory:
`tectonic fig2_duncan.tex`. Every plot has its own CSV inputs here; complete
uncertainty intervals are in `../results/`.

- Fig. 1: conditional Floquet thresholds. The treatment-name to amplitude mapping
  is inferred; host values are phase averages of sparse terminal counts. These are
  not measured growth losses. The clocks are illustrative.
- Fig. 2: replicate means, with no smoothing. The registered growth window ends
  at day 16; day 56 is a different, late endpoint. Constant extreme-temperature
  controls do not share the mean temperature of the variable arms.
- Fig. 3a: primary infection on day 11, secondary infection after day 18.
  Exact binomial intervals are in `s3_threshold.csv`.
- Fig. 3b: the horizontal axis denotes different schedule quantities: full period
  for the square wave, mean dwell per state for the telegraph. The scenario mixes
  host species and is not a prediction validated against a fluctuation experiment.
- Fig. 4: air temperature, not lake water temperature. Source coordinates remain
  approximate. Laboratory outcomes are not matched to outdoor weather.
''')
print('Wrote',len(list(F.glob('*.tex'))),'standalone TeX figures and their CSV inputs.')
