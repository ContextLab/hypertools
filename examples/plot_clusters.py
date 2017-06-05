# -*- coding: utf-8 -*-
"""
=============================
Discovering clusters
=============================

The `n_clusters` kwarg can be used to discover clusters in your dataset.  It
relies on scikit-learn's implementation of k-mean clustering to fin clusters,
and then labels the points accordingly. You must set the number of clusters
yourself.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import pandas as pd

# load example data
data = hyp.tools.load('mushrooms')

# plot
hyp.plot(data, '.', n_clusters=10)
