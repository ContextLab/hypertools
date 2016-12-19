from scipy.linalg import toeplitz
import numpy as np
from copy import copy
import hypertools as hyp

K = 10 - toeplitz(np.arange(10))

data1 = np.cumsum(np.random.multivariate_normal(np.zeros(10), K, 250), axis=0)
data2 = copy(data1)
sample_inds = np.random.choice(data2.shape[0], data2.shape[0]*.1)
for sample in sample_inds:
    data2[sample,:]=np.nan

print(data2[sample,0])
hyp.plot([data1,data2])
