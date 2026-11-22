# -*- coding: utf-8 -*-
"""
=================================
Plotting data with missing values
=================================

When a dataset contains missing (NaN) entries, `hyp.plot` fills them in
before reducing and plotting, using probabilistic principal components
analysis (PPCA) by default. Here a random walk through 10 dimensions is
generated, some of its entries are removed, and the original and imputed
versions are plotted together. The first figure lets the imputation happen
implicitly inside `hyp.plot`; the second uses `hyp.tools.missing_inds` to
find the rows that contained missing values and marks them with stars, so
you can see exactly which points were interpolated.
"""

# Code source: Andrew Heusser
# License: MIT

from copy import copy

import numpy as np
from scipy.linalg import toeplitz

import hypertools as hyp

# simulate a 10-D random walk (seeded so the figures are reproducible)
np.random.seed(123)
K = 10 - toeplitz(np.arange(10))
data1 = np.cumsum(np.random.multivariate_normal(np.zeros(10), K, 250), axis=0)


def remove_entries(data, fraction):
    """Return a copy of `data` with `fraction` of its entries set to NaN."""
    missing = copy(data)
    n_missing = int(missing.size * fraction)
    flat = np.random.choice(missing.size, n_missing, replace=False)
    rows, cols = np.unravel_index(flat, missing.shape)
    missing[rows, cols] = np.nan
    return missing


# %%
# Implicit imputation: remove 10% of the entries and plot both versions.
# `hyp.plot` fills the NaNs with PPCA on its way to the figure, so the dotted
# (imputed) trajectory tracks the solid (original) one closely.
data2 = remove_entries(data1, 0.10)
hyp.plot([data1, data2], linestyle=['-', ':'], legend=['Original', 'PPCA'])

# %%
# Marking the interpolated points: remove 5% of the entries, reduce both
# datasets to 3-D (which imputes the missing values), then use
# `hyp.tools.missing_inds` to pull out the rows that had missing values and
# draw them as stars on top of the two trajectories.
data3 = remove_entries(data1, 0.05)
data1_r, data3_r = hyp.reduce([data1, data3], ndims=3)

missing_inds = hyp.tools.missing_inds(data3)
missing_points = data3_r[missing_inds, :]

hyp.plot([data1_r, data3_r, missing_points], ['-', '--', '*'],
         legend=['Full', 'Missing', 'Missing Points'])
