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

from sklearn import datasets
import hypertools as hyp

digits = datasets.load_digits(n_class=6)
data = digits.data
group = digits.target

hyp.plot(data, 'o', group=group)
