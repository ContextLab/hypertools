# -*- coding: utf-8 -*-
"""
=============================
The Datasaurus Dozen
=============================

The "Datasaurus Dozen" (Matejka & Fitzmaurice, 2017) is a set of 13
datasets that share nearly identical summary statistics (means,
standard deviations, and correlations) but look wildly different when
plotted.  `hyp.load('datasaurus')` returns the datasets as a list of
pandas DataFrames; here we plot *all thirteen* side by side as 2D
scatter plots of small black dots (the ``.`` point marker) to show why
it always pays to visualize your data.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import math

import matplotlib.pyplot as plt

import hypertools as hyp

datasets = hyp.load('datasaurus')

ncols = 4
nrows = math.ceil(len(datasets) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
axes = axes.ravel()

for i, (df, ax) in enumerate(zip(datasets, axes)):
    # small black dots via the '.' point marker
    hyp.plot(df, '.', color='k', ndims=2, ax=ax, title=f'Dataset {i + 1}')
    print(f'Dataset {i + 1}: mean=({df.x.mean():.2f}, {df.y.mean():.2f}), '
          f'sd=({df.x.std():.2f}, {df.y.std():.2f}), '
          f'r={df.x.corr(df.y):.2f}')

# hide any unused panels
for ax in axes[len(datasets):]:
    ax.set_visible(False)

plt.tight_layout()
