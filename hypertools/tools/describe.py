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
from .._shared.helpers import format_data

def describe(x, reduce_model=None, reduce_params=None, max_dims=None, show=True):
    """
    Create plot describing covariance with as a function of number of dimensions

    This function correlates the raw data with reduced data to get a sense
    for how well the data can be summarized with n dimensions.  Useful for
    evaluating quality of dimensionality reduced plots.

    Parameters
    ----------
    x : Numpy array, DataFrame or list of arrays/dfs
        A list of Numpy arrays or Pandas Dataframes

    Returns
    ----------
    fig, ax, result : maplotlib.Figure, matplotlib.Axes, dict
        By default, a matplotlib figure and axis handle, and a data
        dictionary with the analysis results. 'average' is the correlation
        by number of components for all data. 'individual' is a list of lists,
        where each list is a correlation by number of components vector (for each
        input list). If show=False, only result is returned.

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
        alldims = pdist(x,'correlation')


        corrs=[]
        for dims in range(2, max_dims):
            reduced = pdist(reducer(x, ndims=dims, model=reduce_model,
                    model_params=reduce_params),'correlation')
            corrs.append(np.corrcoef(alldims, reduced)[0][1])
            del reduced
        return corrs

    # common format
    x = format_data(x)

    # if max dims is not set, make it the length of the minimum number of columns
    if max_dims is None:
        max_dims = np.min(map(lambda xi: xi.shape[1], x))

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
        return fig, ax, result
    else:
        return result
