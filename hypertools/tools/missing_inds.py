#!/usr/bin/env python

##PACKAGES##
import numpy as np
from .._shared.helpers import format_data

def missing_inds(x):
    """
    Returns indices of missing data

    This function is useful to identify rows of your array that contain missing
    data or nans.  The returned indices can be used to remove the rows with
    missing data, or label the missing data points that are interpolated
    using PPCA.

    Parameters
    ----------
    x : array or list of arrays

    Returns
    ----------
    inds : list, or list of lists
        A list of indices representing rows with missing data. If a list of
        numpy arrays is passed, a list of lists will be returned.

    """

    x = format_data(x)

    inds = [[idx for idx,row in enumerate(arr) if any(np.isnan(row))] for arr in x]

    if len(inds) > 1:
        return inds
    else:
        return inds[0]
