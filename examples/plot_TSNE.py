# -*- coding: utf-8 -*-
"""
=============================
Visualizing the digits dataset using t-SNE
=============================

This example loads in some data from the scikit-learn digits dataset and plots
it using t-SNE.
"""

# Code source: Andrew Heusser
# License: MIT

from sklearn import datasets
import hypertools as hyp

digits = datasets.load_digits(n_class=6)
data = digits.data
hue = digits.target.astype('str')

hyp.plot(data, '.', reduce='TSNE', hue=hue, ndims=2)
