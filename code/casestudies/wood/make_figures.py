#!/usr/bin/env python3
"""Write standalone pgfplots sources and plotting CSVs; no matplotlib needed."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
import numpy as np
F=a.HERE/'figures';F.mkdir(exist_ok=True)
m=json.loads((a.RES/'meta_models.json').read_text())['M7_joint_tax']['REML']
labels=['Fluctuating transfers','Storage stage','Host spreading','Active transfers']
rows=[{'trait':a.TRAITS[i],'y':4-i,'estimate':m['beta'][i+3],'lower':m['ci_lo'][i+3],'upper':m['ci_hi'][i+3]} for i in range(4)]
a.pd.DataFrame(rows).to_csv(F/'trait_effects.csv',index=False)
coords=' '.join(f"({r['estimate']:.8f},{r['y']}) += ({r['upper']-r['estimate']:.8f},0) -= ({r['estimate']-r['lower']:.8f},0)" for r in rows)
tex=r'''\documentclass[tikz,border=8pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[width=13cm,height=7cm,xmin=-0.45,xmax=0.55,ymin=0.5,ymax=4.5,
 xlabel={Trait coefficient (log count per 30.268-year SD)},
 ytick={1,2,3,4},yticklabels={Active transfers,Host spreading,Storage stage,Fluctuating transfers},
 grid=major,title={Registered joint meta-regression: estimates and 95\% intervals}]
\addplot[gray,dashed] coordinates {(0,0.5) (0,4.5)};
\addplot[only marks,mark=*,blue,error bars/.cd,x dir=both,x explicit] coordinates {
'''+coords+r'''};
\end{axis}
\end{tikzpicture}
\end{document}
'''
(F/'trait_effects.tex').write_text(tex)
g=json.loads((a.RES/'reproduction_gate.json').read_text());sd=json.loads((a.RES/'validation.json').read_text())['year_sd'];b=np.array(g['slopes']);se=np.array(g['slope_se']);rate=100*np.expm1(10*b/sd);lo=100*np.expm1(10*(b-1.96*se)/sd);hi=100*np.expm1(10*(b+1.96*se)/sd)
r=[{'host_group':str(i+1) if i<2 else '3plus','x':i+1,'percent_per_decade':rate[i],'lower':lo[i],'upper':hi[i]} for i in range(3)]
a.pd.DataFrame(r).to_csv(F/'host_number_decline.csv',index=False)
coords=' '.join(f"({z['x']},{z['percent_per_decade']:.8f}) += (0,{z['upper']-z['percent_per_decade']:.8f}) -= (0,{z['percent_per_decade']-z['lower']:.8f})" for z in r)
(F/'host_number_decline.tex').write_text(r'''\documentclass[tikz,border=8pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}\begin{tikzpicture}
\begin{axis}[width=10cm,height=7cm,xmin=0.5,xmax=3.5,xtick={1,2,3},xticklabels={1,2,3+},xlabel={Obligate hosts},ylabel={Change per decade (\%)},title={Reproduction: estimates and 95\% Wald intervals},grid=major]
\addplot[gray,dashed] coordinates {(0.5,0) (3.5,0)};
\addplot[only marks,mark=*,blue,error bars/.cd,y dir=both,y explicit] coordinates {
'''+coords+r'''};
\end{axis}\end{tikzpicture}\end{document}
''')
weather=a.pd.read_csv(a.RES/'ersst_annual.csv');weather=weather[(weather.grid_lat==48)&(weather.grid_lon==-124)];weather.to_csv(F/'regional_environment.csv',index=False)
(F/'PLOTTING.md').write_text('''# Plotting notes

`trait_effects.csv`: plot estimate against y, horizontal interval [lower,upper], vertical reference at zero. The registered four traits are in the original full joint meta-regression; intervals use REML and t with residual degrees of freedom. These are not taxon-specific causal effects.

`host_number_decline.csv`: plot percent_per_decade against x, vertical interval [lower,upper], reference at zero. Back-transform the fitted group slope and normal Wald limits using the original collection-year standard deviation.

`regional_environment.csv`: plot year against sst_mean (Celsius) and separately sst_monthly_sd (Celsius). Values are annual summaries of monthly ERSSTv6 at 48 N, 124 W; the SD measures seasonality, not daily temperature variability. This cell is offshore from all collection sites.

The two `.tex` files are self-contained pgfplots figures. Compile with tectonic in this directory, or plot the CSVs with any standard plotting software. No outcome-selected smoothing is used.
''')
a.save('figure_numbers',{'host_number_plot':r,'trait_plot':rows});print(json.dumps(a.plain({'host_number_plot':r,'trait_plot':rows}),indent=2))
