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
import warnings
import numpy as np
from ppca import PPCA
from sklearn.decomposition import PCA as PCA

def reducePCA(x, ndim):
	if np.isnan(np.vstack(x)).any():
		warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')
		x_split= np.cumsum([i.shape[0] for i in x][:-1])
		m = PPCA(np.vstack(x))
		m.fit(d=ndim)
		x_pca = m.transform()
		return list(np.split(x_pca,x_split,axis=0))
	else:
		m=PCA(n_components=ndim, whiten=True)
		m.fit(np.vstack(x))
		return [m.transform(i) for i in x]

##MAIN FUNCTION##
def reduce(arr,ndims=3, method='PCA'):
    if method=='PCA':
        return reducePCA(arr,ndims)
