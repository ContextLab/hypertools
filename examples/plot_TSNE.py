# -*- coding: utf-8 -*-
"""
=============================
Visualizing the digits dataset using t-SNE
=============================

This example loads in some data from the scikit-learn digits dataset and plots
it using t-SNE. Other supported models include PCA, FastICA, MDS, Isomap,
SpectralEmbedding and LocallyLinearEmbedding

"""

# Code source: Andrew Heusser
# License: MIT

from sklearn import datasets
import hypertools as hyp

digits = datasets.load_digits(n_class=6)
data = digits.data
group = digits.target

hyp.plot(data, '.', model='TSNE', group=group, ndims=2, legend=True)
