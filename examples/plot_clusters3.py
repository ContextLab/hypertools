# -*- coding: utf-8 -*-
"""
=============================
Discovering clusters
=============================

To make use of HDBSCAN as the clustering algorithm used to discver clusters
you must specify it as the cluster argument. If you wish to specify HDBSCAN
parameters you will need the dictionary form which includes both the model
and the params. Since HDBSCAN does not require the number of clusters this
value to can set to 0. Note that n_clusters much be set to ensure that
clustering is performed.
"""

# Code source: Andrew Heusser and Leland McInnes
# License: MIT

# import
import hypertools as hyp
import pandas as pd

# load example data
data = hyp.load('mushrooms')

# plot
hyp.plot(data, '.', cluster={'model':'HDBSCAN',
                             'params': {'min_samples':5,
                                        'min_cluster_size':30}},
         n_clusters=0)
