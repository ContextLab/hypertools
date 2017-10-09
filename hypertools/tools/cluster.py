#!/usr/bin/env python
<<<<<<< HEAD
import warnings
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration, SpectralClustering
import numpy as np
from .._shared.helpers import *

@memoize
def cluster(x, cluster='KMeans', n_clusters=3, ndims=None):
    """
    Performs clustering analysis and returns a list of cluster labels
=======

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from .._shared.helpers import *

def cluster(x, n_clusters=8, ndims=None):
    """
    Performs k-means clustering and returns a list of cluster labels
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

    Parameters
    ----------
    x : A Numpy array, Pandas Dataframe or list of arrays/dfs
        The data to be clustered.  You can pass a single array/df or a list.
        If a list is passed, the arrays will be stacked and the clustering
        will be performed across all lists (i.e. not within each list).

<<<<<<< HEAD
    cluster : str or dict
        Model to use to discover clusters.  Support algorithms are: KMeans,
        MiniBatchKMeans, AgglomerativeClustering, Birch, FeatureAgglomeration,
        SpectralClustering (default: KMeans).Can be passed as a string, but for
        finer control of the model parameters, pass as a dictionary, e.g.
        reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}. See
        scikit-learn specific model docs for details on parameters supported for
        each model.

    n_clusters : int
        Number of clusters to discover

    ndims : None
        Deprecated argument.  Please use new analyze function to perform
        combinations of transformations
=======
        n_clusters : int
        The number of clusters to discover (i.e. k)

    ndims : int or None
        This parameter allows you to first reduce dimensionality before
        running k-means
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

    Returns
    ----------
    cluster_labels : list
        An list of cluster labels

    """

<<<<<<< HEAD
    # if cluster is None, just return data
    if cluster is None:
        return x
    else:

        if ndims is not None:
            warnings.warn('The ndims argument is now deprecated. Ignoring dimensionality reduction step.')

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

        # initialize model
        model = model(**model_params)

        # fit the model
        model.fit(np.vstack(x))

        # return the labels
        return list(model.labels_)
=======
    x = format_data(x)

    if type(x) is list:
        x = np.vstack(x)
    if ndims:
        x = PCA(n_components=ndims).fit_transform(x)

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(x)

    return list(kmeans.labels_)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
