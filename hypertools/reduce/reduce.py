#!/usr/bin/env python

import warnings
import numpy as np
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA, MiniBatchSparsePCA, DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap
from .._shared.helpers import *
from ..tools.format_data import format_data as formatter

# dictionary of models
models = {
    'PCA': PCA,
    'IncrementalPCA': IncrementalPCA,
    'SparsePCA': SparsePCA,
    'MiniBatchSparsePCA': MiniBatchSparsePCA,
    'KernelPCA': KernelPCA,
    'FastICA': FastICA,
    'FactorAnalysis': FactorAnalysis,
    'TruncatedSVD': TruncatedSVD,
    'DictionaryLearning': DictionaryLearning,
    'MiniBatchDictionaryLearning': MiniBatchDictionaryLearning,
    'TSNE': TSNE,
    'Isomap': Isomap,
    'SpectralEmbedding': SpectralEmbedding,
    'LocallyLinearEmbedding': LocallyLinearEmbedding,
    'MDS': MDS,
}


def _resolve_model(model_name):
    """Look up a reduction model by name. UMAP is resolved lazily because
    importing umap triggers numba JIT compilation that adds seconds to
    `import hypertools` even when UMAP is never used."""
    if model_name == 'UMAP':
        from umap import UMAP
        return UMAP
    return models[model_name]

# main function
def reduce(x, reduce='IncrementalPCA', ndims=None, internal=False,
           format_data=True):
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

    Returns
    ----------
    x_reduced : Numpy array or list of arrays
        The reduced data with ndims dimensionality is returned.  If the input
        is a list, a list is returned.

    """

    # if model is None, just return data
    if reduce is None:
        return x

    elif isinstance(reduce, str):  # Remove np.string_ check as it's deprecated in NumPy 2.0
        model_name = reduce
        model_params = {
            'n_components': ndims
        }

    elif isinstance(reduce, dict):
        try:
            model_name = reduce['model']
            model_params = reduce['params']
        except KeyError:
            raise ValueError('If passing a dictionary, pass the model as the value of the "model" key and a \
            dictionary of custom params as the value of the "params" key.')

    else:
        # handle other possibilities below
        model_name = reduce

    try:
        # if the model passed is a string, make sure it's one of the supported options
        if isinstance(model_name, str):  # Remove np.string_ check as it's deprecated in NumPy 2.0
            model = _resolve_model(model_name)
        # otherwise check any custom object for necessary methods
        else:
            model = model_name
            getattr(model, 'fit_transform')
            getattr(model, 'n_components')
    except (KeyError, AttributeError):
        raise ValueError('reduce must be one of the supported options or support n_components and fit_transform \
         methods. See http://hypertools.readthedocs.io/en/latest/hypertools.tools.reduce.html#hypertools.tools.reduce \
         for supported models')

    # check for multiple values from n_components & ndims args
    if 'n_components' in model_params:
        if (ndims is None) or (ndims == model_params['n_components']):
            pass
        else:
            warnings.warn('Unequal values passed to dims and n_components. Using ndims parameter.')
            model_params['n_components'] = ndims
    else:
        model_params['n_components'] = ndims

    # convert to common format
    if format_data:
        x = formatter(x, ppca=True)

    # if ndims/n_components is not passed or all data is < ndims-dimensional, just return it
    if model_params['n_components'] is None or all([i.shape[1] <= model_params['n_components'] for i in x]):
        return x

    # Handle empty arrays and type conversion
    stacked_x = np.vstack([np.asarray(arr, dtype=np.float64) for arr in x])
    
    if stacked_x.shape[0] == 1:
        warnings.warn('Cannot reduce the dimensionality of a single row of'
                      ' data. Return zeros length of ndims')
        return [np.zeros((1, model_params['n_components']), dtype=np.float64)]

    elif stacked_x.shape[0] < model_params['n_components']:
            warnings.warn('The number of rows in your data is less than ndims.'
                          ' The data will be reduced to the number of rows.')
            model_params['n_components'] = stacked_x.shape[0]

    # initialize model
    model = model(**model_params)

    # reduce data
    x_reduced = reduce_list(x, model)

    # return data
    if internal or len(x_reduced) > 1:
        return x_reduced
    else:
        return x_reduced[0]


# sub functions
def reduce_list(x, model):
    """Helper function to reduce a list of arrays"""
    # Ensure all arrays are float64 for consistent handling
    x = [np.asarray(arr, dtype=np.float64) for arr in x]
    split = np.cumsum([len(xi) for xi in x])[:-1]
    stacked = np.vstack(x)
    
    # Handle potential NaN values
    if np.any(np.isnan(stacked)):
        warnings.warn('NaN values detected in input data. These may affect the reduction results.')
    
    x_r = np.vsplit(model.fit_transform(stacked), split)
    if len(x) > 1:
        return [xi for xi in x_r]
    else:
        return [x_r[0]]
