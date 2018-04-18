# -*- coding: utf-8 -*-
"""
=============================
Discovering clusters
=============================

The `n_clusters` kwarg can be used to discover clusters in your dataset.  It
relies on scikit-learn's implementation of k-mean clustering to find clusters,
and then labels the points accordingly. You must set the number of clusters
yourself.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
geo = hyp.load('mushrooms')

# plot
geo.plot(n_clusters=10)
