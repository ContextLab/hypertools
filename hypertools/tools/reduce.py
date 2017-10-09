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
def reduce(x, reduce='IncrementalPCA', ndims=None, internal=False):
    """
    Reduces dimensionality of an array, or list of arrays

    Parameters
    ----------
    x : Numpy array or list of arrays
        Dimensionality reduction using PCA is performed on this array.  If
        there are nans present in the data, the function will try to use
        PPCA to interpolate the missing values.

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS. Can be
        passed as a string, but for finer control of the model parameters, pass
        as a dictionary, e.g. reduce={'model' : 'PCA', 'params' : {'whiten' : True}}.
        See scikit-learn specific model docs for details on parameters supported
        for each model.

    ndims : int
        Number of dimensions to reduce

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

    # if model is None, just return data
    if (reduce is None) or (ndims is None):
        return x
    else:

        # common format
        x = format_data(x)

        if all([i.shape[1]<=ndims for i in x]):
            return x

        # if reduce is a string, find the corresponding model
        if type(reduce) in [str, np.string_]:
            model = models[reduce]
            model_params = {
                'n_components' : ndims
            }
        # if its a dict, use custom params
        elif type(reduce) is dict:
            if type(reduce['model']) is str:
                model = models[reduce['model']]
                model_params = reduce['params']
            # if the user specifies a function, set that to the model
            elif callable(reduce['model']):
                model = reduce['model']
                model_params = reduce['params']

        # initialize model
        model = model(**model_params)

        # reduce data
        x_reduced = reduce_list(x, model)

        # return data
        if internal or len(x_reduced)>1:
            return x_reduced
        else:
            return x_reduced[0]
