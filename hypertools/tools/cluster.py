#!/usr/bin/env python

from sklearn.cluster import KMeans
import warnings
import numpy as np
from .._shared.helpers import *
from .reduce import reduce as reducer
from .align import align as aligner
from .normalize import normalize as normalizer

def cluster(x, n_clusters=8, ndims=None, model='IncrementalPCA',
            model_params=None, align=False, normalize=False):
    """
    Performs k-means clustering and returns a list of cluster labels

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a list.
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).

        n_clusters : int
        The number of clusters to discover (i.e. k)

    ndims : int or None
        This parameter allows you to first reduce dimensionality before
        running k-means

    model : str
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS.

    model_params : dict
        Optional dictionary of scikit-learn parameters to pass to reduction model.
        See scikit-learn specific model docs for details.

    align : bool
        If set to True, data will be run through the ``hyperalignment''
        algorithm implemented in hypertools.tools.align (default: False).

    normalize : str or False
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). If set to 'within', the columns will be
        z-scored within each list that is passed. If set to 'row', each row of
        the input data will be z-scored. If set to False, the input data will
        be returned (default is False).

    Returns
    ----------
    cluster_labels : list
        An list of cluster labels

    """

    x = format_data(x)

    if type(x) is list:
        x = np.vstack(x)

    # normalize data
    if normalize:
        x = normalizer(x, normalize=normalize)        

    # reduce data
    if ndims:
        x = reducer(x, ndims=ndims, model=model, model_params=model_params)

    # align data
    if align:
        if len(x) == 1:
            warnings.warn('Data in list of length 1 can not be aligned. '
                          'Skipping the alignment.')
        else:
            x = aligner(x)

    if type(x) is list:
        x = np.vstack(x)
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(x)

    return list(kmeans.labels_)
