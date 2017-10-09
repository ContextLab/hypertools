# -*- coding: utf-8 -*-
"""
=============================
Using the cluster function to label clusters
=============================

Here is an example where we generate some synthetic data, and then use the
cluster function to get cluster labels, which we can then pass to the `group`
kwarg to color our points by cluster.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np
from scipy.stats import multivariate_normal

# simulate clusters
cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+3, np.eye(3), size=100)
data = np.vstack([cluster1, cluster2])

# get cluster labels
cluster_labels = hyp.cluster(data, n_clusters=2)

# plot
hyp.plot(data, '.', group=cluster_labels)
