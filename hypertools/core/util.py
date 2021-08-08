# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np


def get(x, ind, axis=0):
    """
    A robust indexer for iterable objects

    Parameters
    ----------
    :param x: the to-be-indexed object
    :param ind: one or more indices.  If x is not iterable, or if x is empty, this is ignored.  If any(len(x) < ind),
      then the indices wrap around back to the beginning of x, along the indicated axis.
    :param axis: axis to search over (default: 0)

    Returns
    -------
    :return: the selected value(s)
    """

    ind_backup = None
    if not issubclass(type(ind), np.ndarray):
        ind_backup = ind
        ind = np.array(ind)

    if dw.zoo.dataframe_like(x):
        if ind_backup is not None:
            if len(ind.shape) == 0:
                ind = np.array([ind_backup])
        return x.take(ind % x.shape[axis], axis=axis)

    if dw.util.array_like(x) and len(x) > 0:
        if not issubclass(type(x), np.ndarray):
            x = np.array(x)
        return np.take(x, ind % x.shape[axis], axis=axis)
    else:
        return x


def fullfact(dims):
    """
    A Python clone of MATLAB's fullfact function: https://www.mathworks.com/help/stats/fullfact.html

    Parameters
    ----------
    :param dims: a list or array containing the number of "levels" in each condition

    Returns
    -------
    :return: a matrix whose rows correspond to treatments and whose columns correspond to conditions.  The values
      denote, for each treatment, the level for each condition.  All possible combinations of levels (across all
      conditions) are returned.
    """
    vals = np.asmatrix(range(1, dims[0] + 1)).T
    if len(dims) == 1:
        return vals

    aftervals = np.asmatrix(fullfact(dims[1:]))
    inds = np.asmatrix(np.zeros((np.prod(dims), len(dims))))
    row = 0
    for i in range(aftervals.shape[0]):
        inds[row:(row + len(vals)), 0] = vals
        inds[row:(row + len(vals)), 1:] = np.tile(aftervals[i, :], (len(vals), 1))
        row += len(vals)
    return inds


def eval_dict(d):
    for k, v in d.items():
        if type(v) is dict:
            d[k] = eval_dict(v)
        elif type(v) is str:
            d[k] = eval(v)
        else:
            d[k] = v
    return d
