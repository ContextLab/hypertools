# -*- coding: utf-8 -*-
"""
====================
Discovering clusters
====================

Three ways to color a plot by cluster. The first figure passes
``n_clusters`` straight to `hyp.plot`, which runs k-means on the mushrooms
dataset and colors each of the 10 discovered clusters. The second calls
`hyp.cluster` directly on two synthetic blobs to get the labels, then hands
them to ``hue``, which is useful when the labels are needed for something
else too. The third uses the dictionary form of ``cluster`` to run HDBSCAN,
which chooses the number of clusters itself and can mark points as noise
(label -1, colored as their own group). The mushrooms rows are unordered
samples, so they are drawn as points ('.') rather than a connected line.
"""

# Code source: Andrew Heusser and Leland McInnes
# License: MIT

import numpy as np

import hypertools as hyp

# load example data
mushrooms = hyp.load('mushrooms')

# %%
# k-means with a fixed number of clusters, chosen on the plot call.
hyp.plot(mushrooms, '.', n_clusters=10)

# %%
# Labels from `hyp.cluster`, passed to `hue` (seeded so the figure is
# reproducible).
np.random.seed(123)
blob1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
blob2 = np.random.multivariate_normal(np.zeros(3) + 3, np.eye(3), size=100)
blobs = np.vstack([blob1, blob2])

labels = hyp.cluster(blobs, n_clusters=2)
hyp.plot(blobs, '.', hue=labels)

# %%
# HDBSCAN via the dictionary spec: the model name plus its keyword arguments
# (under 'kwargs'). No `n_clusters` is needed.
hyp.plot(mushrooms, '.', cluster={'model': 'HDBSCAN',
                                  'kwargs': {'min_samples': 5,
                                             'min_cluster_size': 30}})
