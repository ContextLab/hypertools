#!/usr/bin/env python

import numpy as np
from .format_data import format_data as formatter


def missing_inds(x, format_data=True):
    """
    Returns indices of missing data

    This function is useful to identify rows of your array that contain missing
    data or nans.  The returned indices can be used to remove the rows with
    missing data, or label the missing data points that are interpolated
    using PPCA.

    Parameters
    ----------
    x : array or list of arrays

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    Returns
    ----------
    inds : 1-D numpy array, None, or list of (1-D numpy array or None)
        For a single array: a 1-D numpy array of the row indices that
        contain missing values, or None when the array has no missing
        data. For a list of arrays: one such entry (array or None) per
        dataset.

    """

    if format_data:
        x = formatter(x, ppca=False)

    inds = []
    for arr in x:
        if np.argwhere(np.isnan(arr)).size == 0:
            inds.append(None)
        else:
            inds.append(np.argwhere(np.isnan(arr))[:,0])
    if len(inds) > 1:
        return inds
    else:
        return inds[0]
