"""Standalone pgfplots plus exact plotting CSVs; no matplotlib dependency."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_io import *
import numpy as np
F=HERE/'figures';F.mkdir(exist_ok=True)
COLORS=['blue','red','teal','orange','violet','black'];TAXA=['small_mammals','birds','fish','freshwater_invertebrates','insects_vectors','arachnid_vectors_ticks']
LABELS=['Small mammals','Birds','Fish','Freshwater invertebrates','Insect vectors','Ticks']
HEAD=r'''\documentclass[tikz,border=8pt]{standalone}
\usepackage{pgfplots}
\pgfdeclarelayer{background}\pgfsetlayers{background,main}
\pgfplotsset{compat=1.18}
\begin{document}
\pagecolor{white}
'''
TAIL='\\end{document}\n'
def tex(name,body):(F/(name+'.tex')).write_text(HEAD+body.replace('\\end{tikzpicture}', r'\begin{pgfonlayer}{background}\fill[white] (current bounding box.south west) rectangle (current bounding box.north east);\end{pgfonlayer}\end{tikzpicture}')+TAIL)
def run():
    rows=read(RESULTS/'memory_regimes.json')['records'];summ=read(RESULTS/'summary_tables.json');sens=read(RESULTS/'sensitivity_tables.json')
    for i,tax in enumerate(TAXA):csvwrite(F/(tax+'_regimes.csv'),[dict(sigma=r['sigma'],sigma2=r['sigma2'],phi=r['r1_bc'],r1=r['r1'],spectral=r['b_spec'],series_id=r['series_id']) for r in rows if r['taxon_group']==tax])
    body=r'\begin{tikzpicture}\begin{axis}[width=15cm,height=9cm,xlabel={Log-abundance standard deviation},ylabel={Corrected one-step memory},xmin=0,xmax=3,ymin=-1,ymax=1,legend style={font=\small,at={(1.02,1)},anchor=north west},grid=major]'+'\n'
    for tax,col,lab in zip(TAXA,COLORS,LABELS):body+=f'\\addplot[only marks,mark=*,mark size=0.8pt,opacity=0.45,{col}] table[x=sigma,y=phi,col sep=comma]{{{tax}_regimes.csv}};\\addlegendentry{{{lab}}}\n'
    body+=r'\addplot[black,dashed,domain=0:3]{0.1919};\addlegendentry{Registered short-stage boundary}'+'\n'+r'\end{axis}\end{tikzpicture}'+'\n';tex('memory_map',body)
    acf=[];sp=[]
    for tax in TAXA:
        for interval in ['annual','subannual']:
            key='taxon_interval:'+tax+':'+interval
            if key not in summ:continue
            g=summ[key]
            for lag in range(1,11):
                v=g['distributions'].get('acf_'+str(lag))
                if not v:continue
                d=v['quantiles']['0.5'];acf.append(dict(taxon=tax,interval=interval,lag=lag,median=d['estimate'],lo=d['ci95'][0],hi=d['ci95'][1],series=g['M'],clusters=g['K']))
            v=g['distributions'].get('b_spec')
            if v:sp.append(dict(taxon=tax,interval=interval,**v['quantiles']['0.5']))
    csvwrite(F/'acf_distributions.csv',acf);csvwrite(F/'spectral_distributions.csv',sp)
    body='\\begin{tikzpicture}\n'
    for n,iv in enumerate(['annual','subannual']):
        body+=f'\\begin{{axis}}[at={{({n*8.5}cm,0)}},anchor=south west,width=8cm,height=7cm,title={{{iv.capitalize()}}},xlabel={{Lag in observation steps}},ylabel={{Median detrended ACF}},xmin=1,xmax=10,ymin=-.6,ymax=.8,legend style={{font=\\tiny,at={{(.5,-.23)}},anchor=north}},grid=major]\n'
        for tax,col,lab in zip(TAXA,COLORS,LABELS):
            ar=[r for r in acf if r['taxon']==tax and r['interval']==iv]
            if not ar:continue
            csvwrite(F/(tax+'_'+iv+'_acf.csv'),ar)
            body+=f'\\addplot[{col},mark=*,mark size=1pt] table[x=lag,y=median,col sep=comma]{{{tax}_{iv}_acf.csv}};\\addlegendentry{{{lab}}}\n'
        body+='\\end{axis}\n'
    body+='\\end{tikzpicture}\n';tex('acf_by_interval',body)
    vals=[]
    for c in ['long','moderate','short']:
        for metric in ['i_half_bound','iii_30pct','iii_10pct']:
            d=summ['overall']['fractions'][c+'_'+metric];vals.append(dict(cycle=c,metric=metric,fraction=d['estimate'],lo=d['ci95'][0],hi=d['ci95'][1],denominator=d['denominator']))
    csvwrite(F/'regime_fractions.csv',vals)
    body=r'\begin{tikzpicture}\begin{axis}[width=13cm,height=8cm,ymin=0,ymax=1.05,ylabel={Fraction of retained host regimes},xtick={1,2,3},xticklabels={Long,Moderate,Short},xlabel={Illustrative adult stage},legend pos=south west,grid=major]'+'\n'
    for metric,col,lab in zip(['i_half_bound','iii_30pct','iii_10pct'],['blue','teal','orange'],['Half-bound buffering','30 percent accuracy','10 percent accuracy']):
        ar=[dict(x=i+1,y=r['fraction'],minus=r['fraction']-r['lo'],plus=r['hi']-r['fraction']) for i,r in enumerate([r for r in vals if r['metric']==metric])];csvwrite(F/(metric+'.csv'),ar)
        body+=f'\\addplot[{col},mark=*,error bars/.cd,y dir=both,y explicit] table[x=x,y=y,y error minus=minus,y error plus=plus,col sep=comma]{{{metric}.csv}};\\addlegendentry{{{lab}}}\n'
    body+=r'\end{axis}\end{tikzpicture}'+'\n';tex('regime_fractions',body)
    forest=[]
    for i,(label,g) in enumerate({'primary':summ['overall'],**sens}.items()):
        d=g['fractions']['short_penalty_zone_T2'];forest.append(dict(row=i,label=label,fraction=d['estimate'],lo=d['ci95'][0],hi=d['ci95'][1],minus=d['estimate']-d['ci95'][0],plus=d['ci95'][1]-d['estimate']))
    csvwrite(F/'sensitivity_fractions.csv',forest)
    ticks=','.join(str(r['row']) for r in forest);labels=','.join(r['label'].replace('_',r'\_') for r in forest)
    body=f'\\begin{{tikzpicture}}\\begin{{axis}}[width=15cm,height=10cm,xmin=0,xmax=1,xlabel={{Fraction above registered memory boundary}},ytick={{{ticks}}},yticklabels={{{labels}}},yticklabel style={{font=\\small}},grid=major]\n'+r'\addplot[only marks,blue,error bars/.cd,x dir=both,x explicit] table[x=fraction,y=row,x error minus=minus,x error plus=plus,col sep=comma]{sensitivity_fractions.csv};'+'\n'+r'\end{axis}\end{tikzpicture}'+'\n';tex('sensitivity_fractions',body)
    extra=sum(r['sigma']>3 for r in rows)
    (F/'plotting_note.txt').write_text('Generated '+now()+'. Run tectonic from this figures directory on each .tex file. CSVs contain exact plotted values. The memory map shows sigma <= 3; '+str(extra)+' extreme series lie outside that viewport and remain in the CSVs and all summaries. ACF lines are group medians, with cluster intervals in acf_distributions.csv. One lag has the physical duration in each record, so annual and subannual panels cannot be pooled as a common clock. Regime error bars resample studies; model simulation error is separate. The sensitivity plot uses the registered short-stage memory boundary, which does not depend on loss interpolation.\n')
    print('figures',len(rows),'out-of-view',extra,flush=True)
if __name__=='__main__':run()
