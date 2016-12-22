import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('weights.mat')
w=[i for i in data['weights'][0][0:3]]

point_colors=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
            tmp.append(int(np.random.randint(10, size=1)))
    point_colors.append(tmp)

hyp.plot(w,'o',point_colors=point_colors)
