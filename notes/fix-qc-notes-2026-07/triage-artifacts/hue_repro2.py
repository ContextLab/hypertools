import os
os.environ['MPLBACKEND']='Agg'
import numpy as np, matplotlib.pyplot as plt
import hypertools as hyp
from hypertools.plot.colors import mat2colors, _get_palette
import seaborn as sns

print('=== controlled k=2 proportion matrix ===')
P = np.array([[0.5,0.5],[0.9,0.1],[0.1,0.9],[0.7,0.3],[0.55,0.45]])
base = np.asarray(_get_palette('hls',2,sns))[:,:3]
print('palette c0',np.round(base[0],3),'c1',np.round(base[1],3))
print('TRUE convex blend (P@base):')
print(np.round(P@base,3))
print('mat2colors output:')
print(np.round(mat2colors(P),3))

print('\n=== step-by-step of mat2colors 2D branch ===')
m=P.astype(float)
w = m - np.min(m,axis=1,keepdims=True)
print('after min-subtract:'); print(np.round(w,3))
rs=w.sum(1,keepdims=True)
w=np.where(rs>0,w/np.where(rs==0,1,rs),0.5)
print('after renormalize:'); print(np.round(w,3))

print('\n=== k=5 matrix (issue 3) ===')
M=np.random.RandomState(1).rand(30,5)
c=mat2colors(M)
print('k=5 mat2colors shape',c.shape,'n unique',len(set(map(tuple,np.round(c,3)))))

print('\n=== color_reduce kwarg exists? ===')
import inspect
sig=inspect.signature(hyp.plot)
print('plot has color_reduce:', 'color_reduce' in sig.parameters)
print('plot has hue:', 'hue' in sig.parameters)
