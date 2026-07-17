# -*- coding: utf-8 -*-
"""
==================================
Discovering clusters using HDBSCAN
==================================

To make use of HDBSCAN as the clustering algorithm used to discover clusters,
you must specify it as the cluster argument. If you wish to specify HDBSCAN
parameters you will need the dictionary form, which includes both the model
name and its keyword arguments (under the 'kwargs' key). Since HDBSCAN does
not require the number of clusters, n_clusters does not need to be set.
Note that HDBSCAN can also label some observations as noise (label -1);
those points are colored as their own group. Because the rows of the
mushrooms dataset are unordered samples, we plot them as points ('.').
"""

# Code source: Andrew Heusser and Leland McInnes
# License: MIT

# import
import hypertools as hyp

# load example data
data = hyp.load('mushrooms')

# plot
hyp.plot(data, '.', cluster={'model': 'HDBSCAN',
                             'kwargs': {'min_samples': 5,
                                        'min_cluster_size': 30}})
