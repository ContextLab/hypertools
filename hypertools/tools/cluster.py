#!/usr/bin/env python

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration, SpectralClustering
import numpy as np
from .._shared.helpers import *

@memoize
def cluster(x, cluster='KMeans', n_clusters=3):
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

    # if cluster is None, just return data
    if cluster is None:
        return x
    else:
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

        # if reduce is a string, find the corresponding model
        if type(cluster) is str:
            model = models[cluster]
            model_params = {
                'n_clusters' : n_clusters
            }
        # if its a dict, use custom params
        elif type(cluster) is dict:
            if type(cluster['model']) is str:
                model = models[cluster['model']]
                model_params = cluster['params']
            # if the user specifies a function, set that to the model
            elif callable(cluster['model']):
                model = cluster['model']
                model_params = cluster['params']

        # initialize model
        model = model(**model_params)

        # fit the model
        model.fit(np.vstack(x))

        # return the labels
        return list(model.labels_)
