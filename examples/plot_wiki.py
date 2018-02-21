# -*- coding: utf-8 -*-
"""
=============================
Plotting a collection of wikipedia pages
=============================

Here, we will plot a collection of wikipedia pages, transformed using a topic
model (the default 'wiki' model) that was fit on the same articles. We will
reduce the dimensionality of the data with UMAP, and then discover cluster with
the 'HDBSCAN' algorithm.

"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
data, labels = hyp.load('wiki')

# plot it
geo = hyp.plot(data, '.', reduce='UMAP', cluster='HDBSCAN')
