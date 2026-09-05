# -*- coding: utf-8 -*-
"""
==============================
Visualizing the digits dataset
==============================

The scikit-learn digits dataset (``hyp.load('digits')``) holds 1797
8x8 grayscale images of handwritten digits, one per row of 64 pixel
columns, plus a ``target`` column naming the digit. Restricting to the
digits 0-5, the three figures below plot the same 64-dimensional pixel
data with three different reducers, colored by digit: the default (PCA)
projection in 3-D, then t-SNE and UMAP in 2-D. The nonlinear reducers pull
each digit into a much tighter, better-separated cluster than the linear
PCA view.
"""

# Code source: Andrew Heusser and Leland McInnes
# License: MIT

import hypertools as hyp

# load the digits as a DataFrame (64 pixel columns + a 'target' column) and
# keep only the digits 0-5
digits = hyp.load('digits')
digits = digits[digits.target < 6]
data = digits.drop(columns='target')

# %%
# The default reducer (PCA) in 3-D. Passing the integer ``target`` column as
# ``hue`` colors each digit as its own group.
hyp.plot(data, '.', hue=digits.target)

# %%
# The same data reduced with t-SNE, in 2-D. String labels give a legend that
# maps each color back to its digit.
hue = digits.target.astype(str)
hyp.plot(data, '.', reduce='TSNE', hue=hue, ndims=2, legend=True)

# %%
# ... and with UMAP, which is faster and keeps more of the global layout.
hyp.plot(data, '.', reduce='UMAP', hue=hue, ndims=2, legend=True)
