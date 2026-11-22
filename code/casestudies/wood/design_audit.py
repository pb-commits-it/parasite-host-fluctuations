"""Recount actual categorical-design VIFs; no outcomes used, no recoding."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
import numpy as np
tr=a.pd.read_csv(a.HERE/'traits_blind.csv');nh=tr.nhosts_wood_3plus.to_numpy();X=np.column_stack([np.ones(len(tr)),nh==2,nh>=3,tr[a.TRAITS]])
labels=['host_2','host_3plus']+a.TRAITS;out={}
for j,label in enumerate(labels,1):
 y=X[:,j];xx=np.delete(X,j,axis=1);r=y-xx@np.linalg.lstsq(xx,y,rcond=None)[0];out[label]=np.sum((y-y.mean())**2)/(r@r)
x={'timestamp':a.stamp(),'rank':int(np.linalg.matrix_rank(X)),'columns':X.shape[1],'n':len(tr),'VIF':out,'correction':'The inherited VIF summary checked a linear host count and a single 3+ indicator separately, not both categorical contrasts together. The power code did use the full categorical design; power outputs are unchanged.'}
a.save('categorical_design_audit',x);print(json.dumps(a.plain(x),indent=2))
