# -*- coding: utf-8 -*-
"""
=============================
Discovering clusters using HDBSCAN
=============================

To make use of HDBSCAN as the clustering algorithm used to discover clusters,
you must specify it as the cluster argument. If you wish to specify HDBSCAN
parameters you will need the dictionary form which includes both the model
and the params. Since HDBSCAN does not require the number of clusters,
n_clusters does not need to be set.
"""

# Code source: Andrew Heusser and Leland McInnes
# License: MIT

# import
import hypertools as hyp
import pandas as pd

# load example data
geo = hyp.load('mushrooms')

# plot
geo.plot(cluster={'model':'HDBSCAN',
                             'params': {'min_samples':5,
                                        'min_cluster_size':30}})
