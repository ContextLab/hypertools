# -*- coding: utf-8 -*-
"""
=============================
The Datasaurus Dozen
=============================

The "Datasaurus Dozen" (Matejka & Fitzmaurice, 2017) is a set of 13
datasets that share nearly identical summary statistics (means,
standard deviations, and correlations) but look wildly different when
plotted.  `hyp.load('datasaurus')` returns the datasets as a list of
pandas DataFrames; here we plot a few of them side by side as 2D
scatter plots to show why it always pays to visualize your data.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import matplotlib.pyplot as plt

import hypertools as hyp

datasets = hyp.load('datasaurus')

# plot the first six frames of the dozen in a 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(9, 6))

for i, ax in enumerate(axes.ravel()):
    df = datasets[i]
    hyp.plot(df, 'o', ndims=2, ax=ax, title=f'Dataset {i + 1}')
    print(f'Dataset {i + 1}: mean=({df.x.mean():.2f}, {df.y.mean():.2f}), '
          f'sd=({df.x.std():.2f}, {df.y.std():.2f}), '
          f'r={df.x.corr(df.y):.2f}')

plt.tight_layout()
plt.show()
