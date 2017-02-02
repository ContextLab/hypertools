#!/usr/bin/env python

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from .._shared.helpers import *

def cluster(x, n_clusters=8, ndims=None):
    """
    Performs k-means clustering and returns a list of cluster labels

    Parameters:
        x : A Numpy array, Pandas Dataframe or list of numpy arrays/dfs
            The data to be clustered.  You can pass a single array/df or a list.
            If a list is passed, the arrays will be stacked and the clustering
            will be performed across all lists (i.e. not within each list).

            n_clusters : int
            The number of clusters to discover (i.e. k)

        ndims : int or None
            This parameter allows you to first reduce dimensionality before
            running k-means

    Returns:
        cluster_labels : list
            An list of cluster labels

    """

    x = format_data(x)

    if type(x) is list:
        x = np.vstack(x)
    if ndims:
        x = PCA(n_components=ndims).fit_transform(x)

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(x)

    return list(kmeans.labels_)
