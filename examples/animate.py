import hypertools as hyp
import scipy.io as sio
import numpy as np
import os

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sample_data/')
data=sio.loadmat(datadir + 'weights.mat')
w=data['weights'][0]
w = [i for i in w]
aligned_w = hyp.tools.align(w)

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

hyp.plot([w1,w2],animate=True)
