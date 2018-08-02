#!/usr/bin/env python
import warnings
import six
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration, SpectralClustering
from .._shared.helpers import *
from .format_data import format_data as formatter

# dictionary of models
models = {
    'KMeans': KMeans,
    'MiniBatchKMeans': MiniBatchKMeans,
    'AgglomerativeClustering': AgglomerativeClustering,
    'FeatureAgglomeration': FeatureAgglomeration,
    'Birch': Birch,
    'SpectralClustering': SpectralClustering,
}

try:
    from hdbscan import HDBSCAN
    _has_hdbscan = True
    models.update({'HDBSCAN': HDBSCAN})
except ImportError:
    _has_hdbscan = False


@memoize
def cluster(x, cluster='KMeans', n_clusters=3, ndims=None, format_data=True):
    """
    Performs clustering analysis and returns a list of cluster labels

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a list.
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).

    cluster : str or dict
        Model to use to discover clusters.  Support algorithms are: KMeans,
        MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration,
        SpectralClustering and HDBSCAN (default: KMeans). Can be passed as a
        string, but for finer control of the model parameters, pass as a
        dictionary, e.g. reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}.
        See scikit-learn specific model docs for details on parameters supported for
        each model.

    n_clusters : int
        Number of clusters to discover. Not required for HDBSCAN.

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    ndims : None
        Deprecated argument.  Please use new analyze function to perform
        combinations of transformations

    Returns
    ----------
    cluster_labels : list
        An list of cluster labels

    """

    if cluster == None:
        return x
    elif (isinstance(cluster, six.string_types) and cluster=='HDBSCAN') or \
    (isinstance(cluster, dict) and cluster['model']=='HDBSCAN'):
        if not _has_hdbscan:
            raise ImportError('HDBSCAN is not installed. Please install hdbscan>=0.8.11')

    if ndims != None:
        warnings.warn('The ndims argument is now deprecated. Ignoring dimensionality reduction step.')

    if format_data:
        x = formatter(x, ppca=True)

    # if reduce is a string, find the corresponding model
    if isinstance(cluster, six.string_types):
        model = models[cluster]
        if cluster != 'HDBSCAN':
            model_params = {
                'n_clusters' : n_clusters
            }
        else:
            model_params = {}
    # if its a dict, use custom params
    elif type(cluster) is dict:
        if isinstance(cluster['model'], six.string_types):
            model = models[cluster['model']]
            model_params = cluster['params']

    # initialize model
    model = model(**model_params)

    # fit the model
    model.fit(np.vstack(x))

    # return the labels
    return list(model.labels_)
