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

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_analyze_thumb.png'

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
geo = hyp.load('weights')
data = geo.get_data()

# process the data
data = hyp.analyze(data, normalize='within', reduce='PCA', ndims=10,
                align='hyper')

# plot it
hyp.plot(data)
