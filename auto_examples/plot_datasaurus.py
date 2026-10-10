# -*- coding: utf-8 -*-
"""
=============================
The Datasaurus Dozen
=============================

The "Datasaurus Dozen" (Matejka & Fitzmaurice, 2017) is a set of 13
datasets that share nearly identical summary statistics (means,
standard deviations, and correlations) but look wildly different when
plotted.  `hyp.load('datasaurus')` returns the datasets as a list of
pandas DataFrames; here we plot *all thirteen* side by side, one panel per
dataset (`panels=True`), as 2D scatter plots of small black dots (the
``.`` point marker) to show why it always pays to visualize your data.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import hypertools as hyp

datasets = hyp.load('datasaurus')

for i, df in enumerate(datasets):
    print(f'Dataset {i + 1}: mean=({df.x.mean():.2f}, {df.y.mean():.2f}), '
          f'sd=({df.x.std():.2f}, {df.y.std():.2f}), '
          f'r={df.x.corr(df.y):.2f}')

# small black dots via the '.' point marker; one panel per dataset, sized
# to a near-square grid, sharing one pipeline fit (a no-op here since
# ndims=2 already matches the data)
fig = hyp.plot(datasets, '.', color='k', ndims=2, panels=True,
               title=[f'Dataset {i + 1}' for i in range(len(datasets))])
