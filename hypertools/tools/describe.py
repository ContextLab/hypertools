#!/usr/bin/env python

##PACKAGES##
from __future__ import division
from builtins import range
import warnings
import numpy as np
from scipy.spatial.distance import pdist
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import seaborn as sns
from .reduce import reduce as reducer
from .._shared.helpers import format_data, memoize

def describe(x, reduce='IncrementalPCA', max_dims=None, show=True):
    """
    Create plot describing covariance with as a function of number of dimensions

    This function correlates the raw data with reduced data to get a sense
    for how well the data can be summarized with n dimensions.  Useful for
    evaluating quality of dimensionality reduced plots.

    Parameters
    ----------

    x : Numpy array, DataFrame or list of arrays/dfs
        A list of Numpy arrays or Pandas Dataframes

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS. Can be
        passed as a string, but for finer control of the model parameters, pass
        as a dictionary, e.g. reduce={'model' : 'PCA', 'params' : {'whiten' : True}}.
        See scikit-learn specific model docs for details on parameters supported
        for each model.

    max_dims : int
        Maximum number of dimensions to consider

    show : bool
        Plot the result (default : true)

    Returns
    ----------

    result : dict
        A dictionary with the analysis results. 'average' is the correlation
        by number of components for all data. 'individual' is a list of lists,
        where each list is a correlation by number of components vector (for each
        input list).

    """

    warnings.warn('When input data is large, this computation can take a long time.')

    def summary(x, max_dims=None):

        # if data is a list, stack it
        if type(x) is list:
            x = np.vstack(x)

        # if no max dims are specified, compute for all of them
        if max_dims is None:
            max_dims = x.shape[1]

        # correlation matrix for all dimensions
        alldims = get_pdist(x)

        corrs=[]
        for dims in range(2, max_dims):
            reduced = get_pdist(reducer(x, ndims=dims, reduce=reduce))
            corrs.append(get_corr(alldims, reduced))
            del reduced
        return corrs

    # common format
    x = format_data(x)

    # if max dims is not set, make it the length of the minimum number of columns
    if max_dims is None:
        max_dims = np.min([xi.shape[1] for xi in x])

    # a dictionary to store results
    result = {}
    result['average'] = summary(x, max_dims)
    result['individual'] = [summary(x_i, max_dims) for x_i in x]

    # if show, plot it
    if show:
        fig, ax = plt.subplots()
        ax = sns.tsplot(data=result['individual'], time=[i for i in range(2, max_dims)], err_style="unit_traces")
        ax.set_title('Correlation with raw data by number of components')
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Number of components')
        plt.show()
    return result

@memoize
def get_corr(reduced, alldims):
    return np.corrcoef(alldims, reduced)[0][1]

@memoize
def get_pdist(x):
    return pdist(x, 'correlation')
