# -*- coding: utf-8 -*-
"""
=============================
Plotting all NIPS papers since 1987
=============================

Here, we will plot a collection of NIPS papers, transformed using a topic
model that was fit on the same articles, and then reduced with UMAP and colored
using the HDBSCAN clustering algorithm. We specify the semantic model
(LatentDirichletAllocation), followed by the corpus we want to use to fit the
model.  We set the corpus to papers to both fit and transform the papers using
the same model.

"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
papers, labels = hyp.load('nips')

# plot it
geo = hyp.plot([papers], '.', semantic='LatentDirichletAllocation', corpus=papers,
               reduce='UMAP', cluster='HDBSCAN')
