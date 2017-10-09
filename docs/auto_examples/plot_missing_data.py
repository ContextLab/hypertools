# -*- coding: utf-8 -*-
"""
=============================
Using the missing_inds function to label interpolated values
=============================

If you have data with missing values, Hypertools will try to interpolate them
using PPCA.  To visualize how well its doing, you can use the missing_inds
function and then highlight the values that were interpolated.  Here, we
generated some synthetic data, removed some values, and then plotted the
original data, data with missing values and highlighted the missing datapoints
with stars.
"""

# Code source: Andrew Heusser
# License: MIT

# import
from scipy.linalg import toeplitz
import numpy as np
from copy import copy
import hypertools as hyp

# simulate data
K = 10 - toeplitz(np.arange(10))
data1 = np.cumsum(np.random.multivariate_normal(np.zeros(10), K, 250), axis=0)
data2 = copy(data1)

# randomly remove 5% of the data
<<<<<<< HEAD
missing = .01
=======
missing = .05
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
inds = [(i,j) for i in range(data1.shape[0]) for j in range(data1.shape[1])]
missing_data = [inds[i] for i in np.random.choice(int(len(inds)), int(len(inds)*missing))]
for i,j in missing_data:
    data2[i,j]=np.nan

# reduce the data
<<<<<<< HEAD
data1_r,data2_r = hyp.reduce([data1, data2], ndims=3)
=======
data1_r,data2_r = hyp.tools.reduce([data1, data2], ndims=3)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

# pull out missing inds
missing_inds = hyp.tools.missing_inds(data2)
missing_data = data2_r[missing_inds, :]

# plot
hyp.plot([data1_r, data2_r, missing_data], ['-', '--', '*'],
         legend=['Full', 'Missing', 'Missing Points'])
