import os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import hypertools as hyp
np.random.seed(0)
A=np.random.randn(60,3)+2; B=np.random.randn(60,3)-2
X=np.vstack([A,B])
soft=np.asarray(hyp.cluster(X,cluster='GaussianMixture',n_clusters=2))
h=np.linspace(0,1,120)

for label,hue in [('matrix',soft),('continuous',h)]:
    fig=hyp.plot(X,'.',ndims=3,hue=hue,backend='plotly')
    print(f'--- {label} hue, plotly ---  type',type(fig).__name__,'ntraces',len(fig.data))
    allc=[]
    for tr in fig.data:
        mc=getattr(getattr(tr,'marker',None),'color',None)
        if mc is not None and not isinstance(mc,str) and hasattr(mc,'__len__'):
            allc.extend(list(mc))
    print('   per-point marker colors:',len(allc),'unique',len(set(map(str,allc))))
    if allc: print('   sample',allc[:2])
print('DONE')
