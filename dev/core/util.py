# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np


def get(x, ind, axis=0):
    if dw.util.array_like(x) and len(x) > 0:
        if not dw.zoo.is_array(x):
            x = np.array(x)
        return np.take(x, ind % x.shape[axis], axis=axis)
    else:
        return x


def fullfact(dims):
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
