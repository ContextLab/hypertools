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
group = digits.target.astype('str')

<<<<<<< HEAD
hyp.plot(data, '.', reduce='TSNE', group=group, ndims=2)
=======
hyp.plot(data, '.', model='TSNE', group=group, ndims=2)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
