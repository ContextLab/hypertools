import plot_coords as coords
import scipy.io as sio
import numpy as np

data=sio.loadmat('examples/weights.mat')
w=data['weights'][0][0:3]

point_colors=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
        if iidx==0:
            tmp.append(np.random.rand())
        else:
            tmp.append(np.random.rand())
    point_colors.append(tmp)

coords.plot_coords(w,'o',point_colors=point_colors)
