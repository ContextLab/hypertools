#!/usr/bin/env python

# libraries
import warnings
import numpy as np

## reduction models
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA, MiniBatchSparsePCA, DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap

# internal libraries
from ..tools.df2mat import df2mat
from .._shared.helpers import *

# main function
@memoize
def reduce(x, ndims=3, model='IncrementalPCA', model_params=None, internal=False):
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
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS.

    model_params : dict
        Optional dictionary of scikit-learn parameters to pass to reduction model.
        See scikit-learn specific model docs for details.

    Returns
    ----------
    x_reduced : Numpy array or list of arrays
        The reduced data with ndims dimensionality is returned.  If the input
        is a list, a list is returned.

    """

    # sub functions
    def reduce_list(x, model):
        split = np.cumsum([len(xi) for xi in x])[:-1]
        x_r = np.vsplit(model.fit_transform(np.vstack(x)), split)
        if len(x)>1:
            return [xi for xi in x_r]
        else:
            return [x_r[0]]

    # dictionary of models
    models = {
        'PCA' : PCA,
        'IncrementalPCA' : IncrementalPCA,
        'SparsePCA' : SparsePCA,
        'MiniBatchSparsePCA' : MiniBatchSparsePCA,
        'KernelPCA' : KernelPCA,
        'FastICA' : FastICA,
        'FactorAnalysis' : FactorAnalysis,
        'TruncatedSVD' : TruncatedSVD,
        'DictionaryLearning' : DictionaryLearning,
        'MiniBatchDictionaryLearning' : MiniBatchDictionaryLearning,
        'TSNE' : TSNE,
        'Isomap' : Isomap,
        'SpectralEmbedding' : SpectralEmbedding,
        'LocallyLinearEmbedding' : LocallyLinearEmbedding,
        'MDS' : MDS
    }

    # common format
    x = format_data(x)

    # if model is None, just return data
    if (model is None) or (ndims is None) or (all([i.shape[1]<=ndims for i in x])):
        return x
    else:

        assert all([i.shape[1]>=ndims for i in x]), "In order to reduce the data, ndims must be less than the number of dimensions"

        # build model params dict
        if model_params is None:
            model_params = {
                'n_components' : ndims
            }
        elif 'n_components' in model_params:
            pass
        else:
            model_params['n_components']=ndims

        # intialize the model instance
        if callable(model):
            model = model(**model_params)
        else:
            model = models[model](**model_params)

        # reduce data
        x_reduced = reduce_list(x, model)

        # return data
        if internal or len(x_reduced)>1:
            return x_reduced
        else:
            return x_reduced[0]
