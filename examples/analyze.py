# -*- coding: utf-8 -*-
"""
=============================
Analyze data and then plot
=============================

This example demonstrates how to use the `analyze` function to process data
prior to plotting. The data is a list of numpy arrays representing
multi-voxel activity patterns (columns) over time (rows).  First, analyze function
normalizes the columns of each matrix (within each matrix). Then the data is
reduced using PCA (10 dims) and finally it is aligned with hyperalignment. We can
then plot the data with hyp.plot, which further reduces it so that it can be
visualized.
"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
data = hyp.tools.load('weights')

# process the data
data = hyp.analyze(data, normalize='within', reduce_model='PCA', ndims=10,
                align_model='hyper')

# plot it
hyp.plot(data)
