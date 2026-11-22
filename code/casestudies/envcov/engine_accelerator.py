"""Sparse acceleration of Engine._products, checked against its original routine."""
from pathlib import Path
import ctypes, subprocess
import numpy as np
import envcov_engine as E
HERE=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
_LIB=None

def setup():
 global _LIB
 src=HERE/'engine_products.c'; so=HERE/'engine_products.so'
 if not so.exists() or so.stat().st_mtime<src.stat().st_mtime:
  subprocess.run(['cc','-O3','-shared','-fPIC',str(src),'-o',str(so)],check=True,capture_output=True)
 _LIB=ctypes.CDLL(str(so))
 _LIB.products.argtypes=[ctypes.c_int]*7+[ctypes.c_void_p]*10
 _LIB.products.restype=ctypes.c_int

def fast_products(self,rates,y,burn,pairs):
 if _LIB is None:setup()
 A=self.base_parts[0]+np.exp(y)*self.base_parts[1]
 Ay=np.exp(y)*self.base_parts[1]
 P=self.rate_parts[0]+np.exp(y)*self.rate_parts[1]
 Py=np.exp(y)*self.rate_parts[1]
 mask=(A!=0)|(Ay!=0)|np.any(P!=0,axis=0)|np.any(Py!=0,axis=0)
 dest,src=np.nonzero(mask)
 arrays=[np.ascontiguousarray(dest,dtype=np.int32),np.ascontiguousarray(src,dtype=np.int32)]
 arrays += [np.ascontiguousarray(z,dtype=np.float64) for z in [A[mask],Ay[mask],P[:,mask],Py[:,mask],rates,self.v]]
 logs=np.zeros(rates.shape[1]);slopes=logs.copy();arrays += [logs,slopes]
 err=_LIB.products(len(rates),rates.shape[1],self.n,len(dest),rates.shape[2],burn,self.period,*[z.ctypes.data for z in arrays])
 if err:raise E.ModelError(f'sparse product failed: {err}')
 if pairs:logs=(logs[:pairs]+logs[pairs:])/2;slopes=(slopes[:pairs]+slopes[pairs:])/2
 return logs,slopes

def install(check=True):
 setup()
 if check:
  descs=[E.make_borrelia_desc(2.,5.,.6,.3),E.make_borrelia_desc(3.,15.,.6,.3,tick_shape='chain',tick_substages=5),E.make_borrelia_desc(2.,5.,.6,.3,mouse_shape='fixed',tick_shape='fixed'),E.make_borrelia_desc(2.,5.,.6,.3,sigma_weather=.2,phi_weather=.2,rho_hw=.5),E.make_borrelia_desc(2.,5.,.6,.3,n_patches=4,patch_move=.15)]
  maxerr=0.
  for d in descs:
   e=E.Engine(d).critical();h=e.environment.sample(240,4,20261115,antithetic=True);rates=e.rates(h)
   for y in [0.,.15]:
    a=e._products(rates,y,40,4);b=fast_products(e,rates,y,40,4)
    maxerr=max(maxerr,max(float(np.max(abs(x-z))) for x,z in zip(a,b)))
  assert maxerr<1e-10,maxerr
  print(f'Accelerator: 10 comparisons with unmodified engine; maximum absolute error {maxerr:.3g}; tolerance 1e-10',flush=True)
 E.Engine._products=fast_products
