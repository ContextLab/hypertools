from scipy.linalg import toeplitz
import numpy as np
from copy import copy
import hypertools as hyp

K = 10 - toeplitz(np.arange(10))

data = np.cumsum(np.random.multivariate_normal(np.zeros(10), K, 250), axis=0)

missing = .1
inds = [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1])]
missing_data = [inds[i] for i in np.random.choice(len(inds), len(inds)*missing)]
for i,j in missing_data:
    data[i,j]=np.nan

missing_inds = hyp.util.get_missing_indices(data)
print(missing_inds)
