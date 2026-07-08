import os
os.environ['MPLBACKEND']='Agg'
import numpy as np, matplotlib.pyplot as plt
import hypertools as hyp
OUT='/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/triage'
np.random.seed(0)
X=np.random.randn(200,3)
h=np.linspace(0,1,200)

print('=== ISSUE 2: surface=True + continuous hue (mpl) ===')
fig=hyp.plot(X,'.',ndims=3,surface=True,hue=h,title='surface+hue')
ax=fig.axes[0]
npts=0; nsurf=0
for c in ax.collections:
    t=type(c).__name__
    fc=c.get_facecolors()
    if 'Path3D' in t or 'PathCollection' in t:
        npts=len(fc); uu=len(set(map(tuple,np.round(fc,3)))) if len(fc) else 0
        print(' scatter pts',npts,'unique colors',uu)
    elif 'Poly3D' in t:
        nsurf+=1; print(' surface Poly3D nfaces',len(fc),'first color',np.round(fc[0],3) if len(fc) else None)
print('scatter present:',npts>0,'surface present:',nsurf>0)
fig.savefig(f'{OUT}/hue_surface_mpl.png',dpi=80); plt.close(fig)

print('\n=== compare: surface=True WITHOUT hue ===')
fig=hyp.plot(X,'.',ndims=3,surface=True,title='surface only')
plt.close(fig)

print('\n=== ISSUE PATTERN: plotly backend matrix hue ===')
hyp.set_interactive_backend('plotly')
A=np.random.randn(60,3)+2; B=np.random.randn(60,3)-2
soft=np.asarray(hyp.cluster(np.vstack([A,B]),cluster='GaussianMixture',n_clusters=2))
figp=hyp.plot(np.vstack([A,B]),'.',ndims=3,hue=soft)
import plotly.graph_objects as go
cols=[]
for tr in figp.data:
    mc=getattr(tr.marker,'color',None)
    if mc is not None and not isinstance(mc,str):
        cols.extend(list(mc))
print('plotly marker colors sample:',cols[:3] if cols else 'NONE (per-trace single color)')
print('plotly n traces:',len(figp.data))
# count distinct marker colors across traces
tcols=[str(getattr(tr.marker,'color',None)) for tr in figp.data]
print('distinct per-trace marker color reprs:',len(set(tcols)))

print('\n=== colorbar=True + matrix hue (mpl) ===')
hyp.set_interactive_backend('matplotlib')
try:
    fig=hyp.plot(np.vstack([A,B]),'.',ndims=3,hue=soft,colorbar=True)
    print('colorbar+matrix hue: OK, axes=',len(fig.axes))
    plt.close(fig)
except Exception as e:
    print('colorbar+matrix hue EXC:',type(e).__name__, str(e)[:200])

print('\n=== colorbar=True + continuous hue (mpl) ===')
try:
    fig=hyp.plot(X,'.',ndims=3,hue=h,colorbar=True)
    print('colorbar+continuous: OK axes=',len(fig.axes)); plt.close(fig)
except Exception as e:
    print('EXC',type(e).__name__,str(e)[:200])
print('DONE')
