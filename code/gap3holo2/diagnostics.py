"""Finite-table growth diagnostics; none is a mathematical convergence test."""
import json,math
from pathlib import Path
import numpy as np
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
x=json.loads((P/'coefficients_K32_P160.json').read_text());x.update(json.loads((P/'coefficients_iid_K60.json').read_text()))
out={}
print('| Host | s | phi | K | a_K | root | factorial root | beta | geom RMS | factorial RMS |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for key,row in x.items():
 a=[float(z) for z in row['coefficients']];K=len(a);k=np.arange(K//2+1,K+1);y=np.log(np.abs(np.array(a)[k-1]));lf=np.array([math.lgamma(t+1) for t in k])
 X=np.array([np.ones(len(k)),k,lf]).T;fit=np.linalg.lstsq(X,y,rcond=None)[0]
 line=np.array([np.ones(len(k)),k]).T
 rms=lambda y:float(np.sqrt(np.mean((y-line@np.linalg.lstsq(line,y,rcond=None)[0])**2)))
 geom=rms(y);fact=rms(y-lf);root=abs(a[-1])**(1/K);broot=math.exp((math.log(abs(a[-1]))-math.lgamma(K+1))/K)
 out[key]={'K':K,'coefficient':a[-1],'ordinary_root':root,'factorial_root':broot,'beta':float(fit[2]),'geometric_fit_RMS':geom,'factorial_fit_RMS':fact}
 host,s,phi=key.split('/');s=s.split('=')[1];phi=phi.split('=')[1]
 print(f'| {host} | {s} | {phi} | {K} | {a[-1]:.5e} | {root:.4f} | {broot:.4f} | {fit[2]:.3f} | {geom:.3f} | {fact:.3f} |')
print('\nGamma iid final ratio a_K/a_(K-1)/(K-1), compared with proposed inverse action:')
for s in ['0.3','0.6','0.9']:
 a=[float(z) for z in x[f'gamma/s={s}/phi=0']['coefficients']];q=float(s);pred=1/(q*(2-q)/(1-q)-2*math.log(1-q));obs=abs(a[-1]/a[-2])/(len(a)-1)
 print(s,obs,pred,'relative deviation',abs(obs/pred-1))
(P/'diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
