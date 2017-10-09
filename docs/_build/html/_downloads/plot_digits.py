# -*- coding: utf-8 -*-
"""
=============================
Visualizing the digits dataset
=============================

This example loads in some data from the scikit-learn digits dataset and plots
it.
"""

# Code source: Andrew Heusser
# License: MIT

# import
from sklearn import datasets
import hypertools as hyp

# load example data
digits = datasets.load_digits(n_class=6)
data = digits.data
group = digits.target

# plot
hyp.plot(data, '.', group=group)
