#!/usr/bin/env python
from __future__ import division
from builtins import range
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from .._shared.helpers import format_data, memoize

@memoize
def normalize(x, normalize='across', internal=False):
    """
    Z-transform the columns or rows of an array, or list of arrays

    This function normalizes the rows or columns of the input array(s).  This
    can be useful because data reduction and machine learning techniques are
    sensitive to scaling differences between features. By default, the function
    is set to normalize 'across' the columns of all lists, but it can also
    normalize the columns 'within' each individual list, or alternatively, for
    each row in the array.

    Parameters
    ----------
    x : Numpy array or list of arrays
        This can either be a single array, or list of arrays

    normalize : str or False or None
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). That is, the z-scores will be computed with
        with repect to column n across all arrays passed in the list. If set
        to 'within', the columns will be z-scored within each list that is
        passed. If set to 'row', each row of the input data will be z-scored.
        If set to False, the input data will be returned with no z-scoring.

    Returns
    ----------
    normalized_x : Numpy array or list of arrays
        An array or list of arrays where the columns or rows are z-scored. If
        the input was a list, a list is returned.  Otherwise, an array is
        returned.

    """

    assert normalize in ['across','within','row', False, None], "scale_type must be across, within, row or none."

    if normalize in [False, None]:
        return x
    else:

        x = format_data(x)

        zscore = lambda X,y: (y - np.mean(X)) / np.std(X) if len(set(y))>1 else np.zeros(y.shape)

        if normalize=='across':
            x_stacked=np.vstack(x)
            normalized_x = [np.array([zscore(x_stacked[:,j], i[:,j]) for j in range(i.shape[1])]).T for i in x]

        elif normalize=='within':
            normalized_x = [np.array([zscore(i[:,j], i[:,j]) for j in range(i.shape[1])]).T for i in x]

        elif normalize=='row':
            normalized_x = [np.array([zscore(i[j,:], i[j,:]) for j in range(i.shape[0])]) for i in x]

        elif normalize==False:
            normalized_x = x

        if internal or len(normalized_x)>1:
            return normalized_x
        else:
            return normalized_x[0]
