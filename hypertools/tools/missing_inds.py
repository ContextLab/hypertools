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
    -------
    inds : 1-D numpy integer array, or list of 1-D numpy integer arrays
        For a single array: a 1-D numpy array of the (unique, sorted) row
        indices that contain missing values -- EMPTY (shape ``(0,)``) when
        the array has no missing data, so downstream fancy indexing like
        ``x[inds, :]`` always yields a well-formed (possibly empty)
        selection. (Returning ``None`` here, as hypertools < 1.0 did,
        made ``x[None, :]`` silently act as ``np.newaxis`` and produce a
        wrong-shaped array.) For a list of arrays: one such entry per
        dataset.

    """

    if format_data:
        x = formatter(x, ppca=False)

    inds = []
    for arr in x:
        hits = np.argwhere(np.isnan(arr))
        if hits.size == 0:
            inds.append(np.array([], dtype=np.intp))
        else:
            inds.append(np.unique(hits[:, 0]))
    if len(inds) > 1:
        return inds
    else:
        return inds[0]
