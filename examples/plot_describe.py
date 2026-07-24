# -*- coding: utf-8 -*-
"""
==============================================================
Using describe to evaluate the integrity of your visualization
==============================================================

The downside to using dimensionality reduction to visualize your data is that
some variance will likely be removed. To help get a sense for the integrity of
your low dimensional visualizations, we built the `describe` function. For
each candidate number of dimensions, it reduces the data and correlates the
pairwise Euclidean distances between observations in the reduced data with
the pairwise distances in the raw (full-dimensional) data, then plots that
correlation as a function of the number of dimensions.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np

# load example data
data = hyp.load('weights_sample')

# plot
hyp.describe(data)
