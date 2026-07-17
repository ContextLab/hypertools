# -*- coding: utf-8 -*-
"""
=============================
Discovering clusters
=============================

The `n_clusters` kwarg can be used to discover clusters in your dataset.  It
relies on scikit-learn's implementation of k-means clustering to find clusters,
and then labels the points accordingly. You must set the number of clusters
yourself. Because the rows of the mushrooms dataset are unordered samples, we
plot them as points (the '.' format string) rather than as a connected line.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
data = hyp.load('mushrooms')

# plot
hyp.plot(data, '.', n_clusters=10)
