#!/usr/bin/env python

##PACKAGES##
import warnings
import numpy as np
from .._externals.ppca import PPCA
from sklearn.decomposition import PCA as PCA
from ..tools.df2mat import df2mat
from ..tools.normalize import normalize as normalizer
from .._shared.helpers import *

##MAIN FUNCTION##
def reduce(x, ndims=3, method='PCA', normalize=False, internal=False):
    """
    Reduces dimensionality of an array, or list of arrays

    Parameters
    ----------
    x : Numpy array or list of arrays
        Dimensionality reduction using PCA is performed on this array.  If
        there are nans present in the data, the function will try to use
        PPCA to interpolate the missing values.

    ndims : int
        Number of dimensions to reduce

    method : str
        Reduction model to use.  Currently, only 'PCA' (PCA/PPCA) is
        implemented. In next release this kwarg will support all scikit-learn
        reduction models.

    normalize : str or False
        Normalizes the data before reducing. If set to 'across', the columns
        of the input data will be z-scored across lists (default). That is,
        the z-scores will be computed with repect to column n across all arrays
        passed in the list. If set to 'within', the columns will be z-scored
        within each list that is passed. If set to 'row', each row of the
        input data will be z-scored. If set to False, the input data will be
        returned with no z-scoring.

    Returns
    ----------
    x_reduced : Numpy array or list of arrays
        The reduced data with ndims dimensionality is returned.  If the input
        is a list, a list is returned.

    """

    ##SUB FUNCTIONS##
    def reducePCA(x, ndim):

        # if there are any nans in any of the lists, use ppca
        if np.isnan(np.vstack(x)).any():
            warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')

            # ppca if missing data
            m = PPCA(np.vstack(x))
            m.fit(d=ndim)
            x_pca = m.transform()

            # if the whole row is missing, return nans
            all_missing = [idx for idx,a in enumerate(np.vstack(x)) if all([type(b)==np.nan for b in a])]
            if len(all_missing)>0:
                for i in all_missing:
                    x_pca[i,:]=np.nan

            # get the original lists back
            if len(x)>1:
                x_split = np.cumsum([i.shape[0] for i in x][:-1])
                return list(np.split(x_pca,x_split,axis=0))
            else:
                return [x_pca]

        else:
            m=PCA(n_components=ndim, whiten=True)
            m.fit(np.vstack(x))
            if len(x)>1:
                return [m.transform(i) for i in x]
            else:
                return [m.transform(x[0])]

    x = format_data(x)

    assert all([i.shape[1]>ndims for i in x]), "In order to reduce the data, ndims must be less than the number of dimensions"

    if normalize:
        x = normalizer(x, normalize=normalize)

    if method=='PCA':
        x_reduced = reducePCA(x,ndims)

    if internal or len(x_reduced)>1:
        return x_reduced
    else:
        return x_reduced[0]
