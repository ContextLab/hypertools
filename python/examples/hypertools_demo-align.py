import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('weights.mat')
w=data['weights'][0]
w = [i for i in w]
aligned_w = hyp.align(w)

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

hyp.plot([w1[:100,:],w2[:100,:]])
