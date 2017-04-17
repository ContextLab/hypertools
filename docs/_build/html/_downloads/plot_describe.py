# -*- coding: utf-8 -*-
"""
=============================
Using describe_pca to evaluate the integrity of your visualization
=============================

The downside to using PCA to visualize your data is that some variance will
likely be removed. To help get a sense for the integrity of your low
dimensional visualizations, we built the `describe_pca` function, which computes
the covariance (samples by samples) of both the raw and reduced datasets, and
plots their correlation.  The function repeats this analysis from 2:N dimensions
until the correlation reaches a local maximum.  Often times this will be less
than the total number of dimensions because the PCA model is whitened.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights_sample')

hyp.tools.describe_pca(data)
