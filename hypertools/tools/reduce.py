#!/usr/bin/env python

import warnings
import numpy as np
import six
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA, MiniBatchSparsePCA, DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap
from umap import UMAP
from .._shared.helpers import *
from .normalize import normalize as normalizer
from .align import align as aligner
from .format_data import format_data as formatter

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
    'MDS' : MDS,
    'UMAP' : UMAP
}

# main function
@memoize
def reduce(x, reduce='IncrementalPCA', ndims=None, normalize=None, align=None,
           model=None, model_params=None, internal=False, format_data=True):
    """
    Reduces dimensionality of an array, or list of arrays

    Parameters
    ----------
    x : Numpy array or list of arrays
        Dimensionality reduction using PCA is performed on this array.

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, MDS and UMAP.
        Can be passed as a string, but for finer control of the model
        parameters, pass as a dictionary, e.g. reduce={'model' : 'PCA',
        'params' : {'whiten' : True}}. See scikit-learn specific model docs
        for details on parameters supported for each model.

    ndims : int
        Number of dimensions to reduce

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    model : None
        Deprecated argument.  Please use reduce.

    model_params : None
        Deprecated argument.  Please use reduce.

    align : None
        Deprecated argument.  Please use new analyze function to perform
        combinations of transformations

    normalize : None
        Deprecated argument.  Please use new analyze function to perform
        combinations of transformations

    Returns
    ----------
    x_reduced : Numpy array or list of arrays
        The reduced data with ndims dimensionality is returned.  If the input
        is a list, a list is returned.

    """

    # deprecated warning
    if (model is not None) or (model_params is not None):
        warnings.warn('Model and model params will be deprecated.  Please use the \
                      reduce keyword.  See API docs for more info: http://hypertools.readthedocs.io/en/latest/hypertools.tools.reduce.html#hypertools.tools.reduce')
        reduce = {}
        reduce['model'] = model
        reduce['params'] = model_params

    # if model is None, just return data
    if reduce is None:
        return x
    else:

        # common format
        if format_data:
            x = formatter(x, ppca=True)

        stacked_x = np.vstack(x)

        if stacked_x.shape[0] == 1:
            warnings.warn('Cannot reduce the dimensionality of a single row of'
                          ' data. Return zeros length of ndims')
            return [np.zeros((1, ndims))]


        elif stacked_x.shape[0] < ndims:
                warnings.warn('The number of rows in your data is less than ndims.'
                              ' The data will be reduced to the number of rows.')

        # deprecation warnings
        if normalize is not None:
            warnings.warn('The normalize argument will be deprecated for this function.  Please use the \
                          analyze function to perform combinations of these transformations.  See API docs for more info: http://hypertools.readthedocs.io/en/latest/hypertools.analyze.html#hypertools.analyze')
            x = normalizer(x, normalize=normalize)

        if align is not None:
            warnings.warn('The align argument will be deprecated for this function.  Please use the \
                          analyze function to perform combinations of these transformations.  See API docs for more info: http://hypertools.readthedocs.io/en/latest/hypertools.analyze.html#hypertools.analyze')
            x = aligner(x, align=align)

        # format params to pass to model
        if isinstance(reduce, dict):
            try:
                model_name = reduce['model']
                model_params = reduce['params']
            except KeyError:
                raise ValueError('If passing a dictionary, pass the model as the value of the "model" key and a \
                dictionary of custom params as the value of the "params" key.')
        else:
            model_name = reduce

        try:
            # if a string is passed, make sure it's one of the supported options
            if isinstance(model_name, (str, np.string_)):
                model = models[model_name]
                model_params = {
                    'n_components': ndims
                }
            # check any custom objects for necessary methods
            else:
                model = model_name
                getattr(model_name, 'fit_transform')
                getattr(model_name, 'n_components')
        except (KeyError, AttributeError):
            raise ValueError('reduce must be one of the supported options or support n_components and fit_transform \
             methods. See http://hypertools.readthedocs.io/en/latest/hypertools.tools.reduce.html#hypertools.tools.reduce \
             for supported models')

        # check for multiple values from n_components & ndims args
        if ndims is not None and 'n_components' in model_params and ndims != model_params['n_components']:
            warnings.warn('Unequal values passed to dims and n_components. Using custom n_components parameter.')

        # if ndims is not passed or all data is < ndims-dimensional, just return it
        if model_params['n_components'] is None or all([i.shape[1] <= model_params['n_components'] for i in x]):
            return x

        # initialize model
        model = model(**model_params)

        # reduce data
        x_reduced = reduce_list(x, model)

        # return data
        if internal or len(x_reduced)>1:
            return x_reduced
        else:
            return x_reduced[0]


# sub functions
def reduce_list(x, model):
    split = np.cumsum([len(xi) for xi in x])[:-1]
    x_r = np.vsplit(model.fit_transform(np.vstack(x)), split)
    if len(x)>1:
        return [xi for xi in x_r]
    else:
        return [x_r[0]]
