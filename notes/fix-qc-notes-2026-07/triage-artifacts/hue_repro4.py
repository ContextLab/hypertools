import os
os.environ['MPLBACKEND']='Agg'
import numpy as np, matplotlib.pyplot as plt
import hypertools as hyp
np.random.seed(0)
X=np.random.randn(200,3); h=np.linspace(0,1,200)

print('=== ISSUE 2b: LINE mode (default, no fmt) surface=True + hue ===')
fig=hyp.plot(X,ndims=3,surface=True,hue=h,title='line surface hue')
ax=fig.axes[0]
nLC=0; nsurf=0; uu=0
for c in ax.collections:
    t=type(c).__name__
    if 'Line3D' in t and len(c.get_colors())>1:
        nLC+=1; uu=len(set(map(tuple,np.round(c.get_colors(),3))))
    if 'Poly3D' in t: nsurf+=1
print('multicolor line segs unique colors:',uu,'nLineColl:',nLC,'surface:',nsurf>0)
print('n ax.lines:',len(ax.lines))
plt.close(fig)

print('\n=== plotly return type ===')
hyp.set_interactive_backend('plotly')
figp=hyp.plot(X,'.',ndims=3,hue=h)
print('type:',type(figp).__module__+'.'+type(figp).__name__)
data=getattr(figp,'data',None)
if data is None and hasattr(figp,'figure'): data=figp.figure.data
print('has .data:',data is not None)
if data is not None:
    for tr in data[:2]:
        mc=getattr(getattr(tr,'marker',None),'color',None)
        print('  trace',tr.type, 'marker.color type', type(mc).__name__, 'sample', (mc[:2] if hasattr(mc,'__len__') and not isinstance(mc,str) else mc))
hyp.set_interactive_backend('matplotlib')
print('DONE')
