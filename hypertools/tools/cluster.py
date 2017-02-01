#!/usr/bin/env python

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from .._shared.helpers import *

def cluster(x, n_clusters=8, ndims=None):
    """
    Aligns a list of arrays

    This function takes a list of high dimensional arrays and 'hyperaligns' them
    to a 'common' space, or coordinate system following the approach outlined by
    Haxby et al, 2011. Hyperalignment uses linear transformations (rotation,
    reflection, translation, scaling) to register a group of arrays to a common
    space. This can be useful when two or more datasets describe an identical
    or similar system, but may not be in same coordinate system. For example,
    consider the example of fMRI recordings (voxels by time) from the visual
    cortex of a group of subjects watching the same movie: The brain responses
    should be highly similar, but the coordinates may not be aligned.

    Haxby JV, Guntupalli JS, Connolly AC, Halchenko YO, Conroy BR, Gobbini
    MI, Hanke M, and Ramadge PJ (2011)  A common, high-dimensional model of
    the representational space in human ventral temporal cortex.  Neuron 72,
    404 -- 416.

    Parameters
    ----------
    data : list
        A list of Numpy arrays or Pandas Dataframes

    method : str
        Either 'hyper' or 'SRM'.  If 'hyper' (default),

    Returns
    ----------
    aligned : list
        An aligned list of numpy arrays

    """

    x = format_data(x)

    if type(x) is list:
        x = np.vstack(x)
    if ndims:
        x = PCA(n_components=ndims).fit_transform(x)

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(x)

    return list(kmeans.labels_)
