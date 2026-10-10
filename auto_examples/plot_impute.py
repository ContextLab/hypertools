# -*- coding: utf-8 -*-
"""
===============================================
Imputing missing data: PPCA vs Kalman smoothing
===============================================

Hypertools fills missing (NaN) values via `hypertools.impute` before
reducing/plotting. This compares two imputers on the `weights_avg` dataset
after randomly knocking out 10% of its entries -- plus three CONSECUTIVE
rows where every feature is missing. That fully-missing-row case is the
case that separates the two imputers: PPCA reconstructs a row from its own
observed features, so a row with NO observed features at all cannot be
recovered, so PPCA warns and leaves those rows NaN (they are dropped below
purely so the PPCA panel has something plottable). The Kalman imputer
instead smooths across time, so it can fill a fully-missing row from the
neighboring (observed) timepoints, at the cost of assuming the data are a
reasonably smooth timeseries -- its panel keeps every row.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

# import
import numpy as np
import hypertools as hyp

# load example data and knock out some values
np.random.seed(42)
data = hyp.load('weights_avg')[0]
missing = data.copy()
n_rows, n_cols = missing.shape

# randomly remove 10% of the entries
n_missing = int(0.1 * n_rows * n_cols)
flat_inds = np.random.choice(n_rows * n_cols, size=n_missing, replace=False)
rows, cols = np.unravel_index(flat_inds, (n_rows, n_cols))
missing[rows, cols] = np.nan

# knock out 3 consecutive fully-missing rows
missing[40:43, :] = np.nan

# impute with PPCA (cannot fill the fully-NaN rows) and Kalman (can)
ppca_imputed = hyp.impute(missing, model='PPCA')
kalman_imputed = hyp.impute(missing, model='Kalman')

# PPCA cannot reconstruct rows with zero observed features, so those 3 rows
# come back as NaN; drop them before plotting (there is nothing left to
# reduce/plot for that timepoint). Kalman fills every row, so nothing to drop.
ppca_plottable = ppca_imputed.dropna(axis=0, how='all')

# plot side by side: one panel per imputed dataset, sharing a single reduce
# fit across both (panels=True)
fig = hyp.plot(
    [ppca_plottable, kalman_imputed], panels=True,
    title=[f'PPCA ({n_rows - len(ppca_plottable)} unfillable rows dropped)',
           'Kalman (all rows filled)'])
