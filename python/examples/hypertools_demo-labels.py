import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('weights.mat')
w=[i for i in data['weights'][0][0:3]]

labels=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
        if iidx==0:
            tmp.append('Subject ' + str(idx))
        else:
            tmp.append(None)
    labels.append(tmp)

hyp.plot(w,'o',labels=labels)
