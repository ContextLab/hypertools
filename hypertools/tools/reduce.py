#!/usr/bin/env python

# libraries
import warnings
import numpy as np

## reduction models
from .._externals.ppca import PPCA
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap

# internal libraries
from ..tools.df2mat import df2mat
from ..tools.normalize import normalize as normalizer
from .._shared.helpers import *

# main function
def reduce(x, ndims=3, model='PCA', model_params={}, normalize=False, internal=False):
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

    model : str
        Decomposition/manifold learning model to use.  Models supported: PCA,
        FastICA, TSNE, MDS, Isomap, SpectralEmbedding, and LocallyLinearEmbedding.
        See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
        for details.

    model_params : dict
        Optional dictionary to pass parameters to model. See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
        for details of each model.

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

    # sub functions
    def fill_missing(x):

        # ppca if missing data
        m = PPCA()
        m.fit(data=np.vstack(x), d=ndim)
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

    def reduce_list(x, model, model_params):
        split = np.cumsum([len(xi) for xi in x])[:-1]
        m=model(**model_params)
        x_r = np.vsplit(m.fit_transform(np.vstack(x)), split)
        if len(x)>1:
            return [xi for xi in x_r]
        else:
            return [x_r[0]]

    # dictionary of models
    models = {
        'PCA' : PCA,
        'FastICA' : FastICA,
        'TSNE' : TSNE,
        'Isomap' : Isomap,
        'SpectralEmbedding' : SpectralEmbedding,
        'LocallyLinearEmbedding' : LocallyLinearEmbedding
    }

    # main
    x = format_data(x)

    assert all([i.shape[1]>ndims for i in x]), "In order to reduce the data, ndims must be less than the number of dimensions"

    # if there are any nans in any of the lists, use ppca
    if np.isnan(np.vstack(x)).any():
        warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')
        x = fill_missing(x)

    # normalize
    if normalize:
        x = normalizer(x, normalize=normalize)

    # build model params dict
    if model_params=={}:
        model_params = {
            'n_components' : ndims
        }
    elif 'n_components' in model_params:
        pass
    else:
        model_params['n_components']=ndims

    # reduce data
    x_reduced = reduce_list(x, models[model], model_params)

    # pad cols with zeros if ndims returned is less than ndims
    if x_reduced[0].shape[1] < ndims:
        for idx, x_r in enumerate(x_reduced):
            x_reduced[idx] = np.hstack([x_r, np.zeros((x_r.shape[0], ndims-x_reduced[0].shape[1]))])

    # return data
    if internal or len(x_reduced)>1:
        return x_reduced
    else:
        return x_reduced[0]
