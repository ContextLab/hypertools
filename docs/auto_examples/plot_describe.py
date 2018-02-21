# -*- coding: utf-8 -*-
"""
=============================
Using describe_pca to evaluate the integrity of your visualization
=============================

The downside to using dimensionality reduction to visualize your data is that
some variance will likely be removed. To help get a sense for the integrity of your low
dimensional visualizations, we built the `describe` function, which computes
the covariance (samples by samples) of both the raw and reduced datasets, and
plots their correlation.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np

# load example data
data, labels = hyp.load('weights_sample')

# plot
hyp.describe(data)
