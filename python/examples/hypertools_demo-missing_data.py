from scipy.linalg import toeplitz
import numpy as np
from copy import copy
import hypertools as hyp

K = 10 - toeplitz(np.arange(10))
data1 = np.cumsum(np.random.multivariate_normal(np.zeros(10), K, 250), axis=0)
data2 = copy(data1)

missing = .05
inds = [(i,j) for i in range(data1.shape[0]) for j in range(data1.shape[1])]
missing_data = [inds[i] for i in np.random.choice(len(inds), len(inds)*missing)]
for i,j in missing_data:
    data2[i,j]=np.nan

data1_r,data2_r = hyp.util.reduce([data1,data2],ndims=3)

missing_inds = hyp.util.missing_inds(data2)
missing_data = data2_r[missing_inds,:]

hyp.plot([data1_r, data2_r, missing_data], ['r','b--','b*'])
