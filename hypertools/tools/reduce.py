#!/usr/bin/env python

# libraries
import warnings
import numpy as np

## reduction models
<<<<<<< HEAD
=======
from .._externals.ppca import PPCA
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA, MiniBatchSparsePCA, DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap

# internal libraries
from ..tools.df2mat import df2mat
<<<<<<< HEAD
from .._shared.helpers import *
from .normalize import normalize as normalizer
from .align import align as aligner

# main function
@memoize
def reduce(x, reduce='IncrementalPCA', ndims=None, normalize=None, align=None,
           model=None, model_params=None, internal=False):
=======
from ..tools.normalize import normalize as normalizer
from .._shared.helpers import *

# main function
def reduce(x, ndims=3, model='IncrementalPCA', model_params={}, normalize=False, internal=False,
           align=False):
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    """
    Reduces dimensionality of an array, or list of arrays

    Parameters
    ----------
    x : Numpy array or list of arrays
<<<<<<< HEAD
        Dimensionality reduction using PCA is performed on this array.

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS. Can be
        passed as a string, but for finer control of the model parameters, pass
        as a dictionary, e.g. reduce={'model' : 'PCA', 'params' : {'whiten' : True}}.
        See scikit-learn specific model docs for details on parameters supported
        for each model.
=======
        Dimensionality reduction using PCA is performed on this array.  If
        there are nans present in the data, the function will try to use
        PPCA to interpolate the missing values.
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

    ndims : int
        Number of dimensions to reduce

<<<<<<< HEAD
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
=======
    model : str
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS.

    model_params : dict
        Optional dictionary of scikit-learn parameters to pass to reduction model.
        See scikit-learn specific model docs for details.

    normalize : str or False
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). If set to 'within', the columns will be
        z-scored within each list that is passed. If set to 'row', each row of
        the input data will be z-scored. If set to False, the input data will
        be returned (default is False).

    align : bool
        If set to True, data will be run through the ``hyperalignment''
        algorithm implemented in hypertools.tools.align (default: False).
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

    Returns
    ----------
    x_reduced : Numpy array or list of arrays
        The reduced data with ndims dimensionality is returned.  If the input
        is a list, a list is returned.

    """

    # sub functions
<<<<<<< HEAD
    def reduce_list(x, model):
        split = np.cumsum([len(xi) for xi in x])[:-1]
        x_r = np.vsplit(model.fit_transform(np.vstack(x)), split)
=======
    def fill_missing(x):

        # ppca if missing data
        m = PPCA()
        m.fit(data=np.vstack(x))
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
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
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

<<<<<<< HEAD
    # deprecated warning
    if (model is not None) or (model_params is not None):
        warnings.warn('Model and model params will be deprecated.  Please use the \
                      reduce keyword.  See API docs for more info: http://hypertools.readthedocs.io/en/latest/hypertools.tools.reduce.html#hypertools.tools.reduce')
        reduce = {}
        reduce['model'] = model
        reduce['params'] = model_params

    # if model is None, just return data
    if (reduce is None) or (ndims is None):
        return x
    else:

        # common format
        x = format_data(x)

        # deprecation warnings
        if normalize is not None:
            warnings.warn('The normalize argument will be deprecated for this function.  Please use the \
                          analyze function to perform combinations of these transformations.  See API docs for more info: http://hypertools.readthedocs.io/en/latest/hypertools.analyze.html#hypertools.analyze')
            x = normalizer(x, normalize=normalize)

        if align is not None:
            warnings.warn('The align argument will be deprecated for this function.  Please use the \
                          analyze function to perform combinations of these transformations.  See API docs for more info: http://hypertools.readthedocs.io/en/latest/hypertools.analyze.html#hypertools.analyze')
            x = aligner(x, align=align)

        # if the shape of the data is already less than ndims, just return it
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
                if reduce['params'] is None:
                    model_params = {
                        'n_components' : ndims
                    }
                else:
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
=======
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

    if align == True:
        # Import is here to avoid circular imports with reduce.py
        from .align import align as aligner
        x_reduced = aligner(x_reduced)

    # return data
    if internal or len(x_reduced)>1:
        return x_reduced
    else:
        return x_reduced[0]
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
