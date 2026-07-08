import os
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib.pyplot as plt
import hypertools as hyp
from hypertools.plot.colors import mat2colors, colors2groups

np.random.seed(0)
OUT = '/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/triage'

A = np.random.randn(80, 3) + np.array([2.2, 0, 0])
B = np.random.randn(80, 3) + np.array([-2.2, 0, 0])
Xc = np.vstack([A, B])

soft = np.asarray(hyp.cluster(Xc, cluster='GaussianMixture', n_clusters=2))
print('soft shape', soft.shape, 'dtype', soft.dtype, 'rowsum0', soft.sum(1)[0])
i_even = int(np.argmin(np.abs(soft[:,0]-0.5)))
i_hi = int(np.argmax(soft[:,0]))
i_lo = int(np.argmin(soft[:,0]))
print('even row', i_even, soft[i_even])
print('hi row', i_hi, soft[i_hi])
print('lo row', i_lo, soft[i_lo])

cols = mat2colors(soft, palette='hls')
print('mat2colors shape', cols.shape)
print('color even', np.round(cols[i_even],4))
print('color hi  ', np.round(cols[i_hi],4))
print('color lo  ', np.round(cols[i_lo],4))
print('n unique mat2colors', len(set(map(tuple, np.round(cols,4)))))
gids,_ = colors2groups(cols)
print('colors2groups n groups', len(set(gids)))

# scatter via hyp.plot
fig = hyp.plot(Xc, '.', ndims=3, hue=soft, title='matrix hue scatter mpl')
ax = fig.axes[0]
fcs=[]
for coll in ax.collections:
    fc = coll.get_facecolors()
    if len(fc): fcs.append(fc)
    print('coll', type(coll).__name__, 'nfc', len(fc))
if fcs:
    allfc=np.vstack(fcs)
    print('scatter total facecolors', allfc.shape, 'unique', len(set(map(tuple,np.round(allfc,3)))))
fig.savefig(f'{OUT}/hue_matrix_scatter_mpl.png', dpi=80); plt.close(fig)

# line mode
fig2 = hyp.plot([A,B], ndims=3, hue=soft, title='matrix hue line mpl')
ax2 = fig2.axes[0]
print('line: n collections', len(ax2.collections), 'n lines', len(ax2.lines))
fig2.savefig(f'{OUT}/hue_matrix_line_mpl.png', dpi=80); plt.close(fig2)
print('DONE')
