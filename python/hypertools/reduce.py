#!/usr/bin/env python

"""
Implements PCA (wrapper for scikit-learn.decomposition.PCA)

INPUTS:
-numpy array(s)
-list of numpy arrays

OUTPUTS:
-numpy array (or list of arrays) with dimensions reduced
"""

##PACKAGES##
import numpy as np
from sklearn.decomposition import PCA as PCA

def reducePCA(x, ndim):
	m=PCA(n_components=ndim, whiten=True)
	m.fit(np.vstack(x))

	r=[]
	for i in x:
		r.append(m.transform(i))
	return r

##MAIN FUNCTION##
def reduce(arr,ndims=3, method='PCA'):
    if method=='PCA':
        return reducePCA(arr,ndims)
