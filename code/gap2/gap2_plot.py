#!/usr/bin/env python3
"""Plot the saved GAP 2 deterministic results; no new numerical experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
rows = json.loads((root/'gap2_results.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
for phi, color in [('-.5','#a44724'),('0','#256393'),('.5','#348164')]:
    rs = [r for r in rows if r['delta']=='.5' and r['phi']==phi]
    L = np.array([r['k']*np.log(10) for r in rs])
    gap = np.array([float(r['gap']) for r in rs])
    A = float(rs[0]['A'])
    axes[0].plot(L, gap, 'o-', ms=4, color=color, label=rf'$\phi={float(phi):g}$')
    axes[0].plot(L, A/L, '--', color=color, lw=1)
    axes[1].plot(1/L, L*gap/A, 'o-', ms=4, color=color)
axes[0].set(xscale='log',yscale='log',xlabel=r'$L_0=\log(1/s)$',
            ylabel=r'$\ell-D(s)$',title='Nonlinear evaluation and leading term')
axes[0].legend(frameon=False)
axes[0].text(.04,.05,'Dashed: proved leading term $A/L_0$',
             transform=axes[0].transAxes, fontsize=9)
axes[1].axhline(1,color='#555555',ls='--',lw=1)
axes[1].set(xlabel=r'$1/L_0$  (limit at left)',ylabel=r'$L_0(\ell-D)/A$',
            title='Slow convergence of the scaled correction')
axes[1].set_xlim(left=0)
for ax in axes:
    ax.grid(alpha=.15)
fig.suptitle(r'GAP 2: $\delta=0.5$, $s=10^{-k}$, $k=4,8,12,18,24,30,40$',fontsize=12)
fig.savefig(root/'gap2_convergence.png')
fig.savefig(root/'gap2_convergence.pdf')
print(root/'gap2_convergence.png')
