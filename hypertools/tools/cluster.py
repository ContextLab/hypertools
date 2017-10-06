#!/usr/bin/env python

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration, SpectralClustering
import numpy as np
from .._shared.helpers import *

@memoize
def cluster(x, model='KMeans', model_params=None, n_clusters=3):
    """
    Performs k-means clustering and returns a list of cluster labels

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a list.
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).

    model : str or function
        Model to use to discover clusters (default: KMeans)

    model_params : dict
        Parameters for the model (default: None)

    Returns
    ----------
    cluster_labels : list
        An list of cluster labels

    """

    x = format_data(x)

    # dictionary of models
    models = {
        'KMeans' : KMeans,
        'MiniBatchKMeans' : MiniBatchKMeans,
        'AgglomerativeClustering' : AgglomerativeClustering,
        'FeatureAgglomeration' : FeatureAgglomeration,
        'Birch' : Birch,
        'SpectralClustering' : SpectralClustering
    }

    # build model params dict
    if model_params is None:
        model_params = {
            'n_clusters' : n_clusters
        }
    elif 'n_clusters' in model_params:
        pass
    else:
        model_params['n_clusters']=n_clusters

    # intialize the model instance
    if callable(model):
        model = model(**model_params)
    else:
        model = models[model](**model_params)

    # fit the model
    model.fit(np.vstack(x))

    # return the labels
    return list(model.labels_)
