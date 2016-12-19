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
from .helpers import *

def reducePCA(x, ndim):
	if np.isnan(np.vstack(x)).any():
		warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')
		x_split= np.cumsum([i.shape[0] for i in x][:-1])
		m = PPCA(np.vstack(x))
		m.fit(d=ndim)
		x_pca = m.transform(np.vstack(x))
		x_pca_interp = interp_col_nans(x_pca)
		return list(np.split(x_pca_interp,x_split,axis=0))
	else:
		m=PCA(n_components=ndim, whiten=True)
		m.fit(np.vstack(x))
		return [m.transform(i) for i in x]

##MAIN FUNCTION##
def reduce(arr,ndims=3, method='PCA'):
    if method=='PCA':
        return reducePCA(arr,ndims)
