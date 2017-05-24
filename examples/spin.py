import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights')
aligned_w = hyp.tools.align(data)

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

hyp.plot([w1,w2], animate='spin')
