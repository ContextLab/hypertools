# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np


@dw.decorate.apply_stacked
def zscore(x, axis=0):
    if axis == 1:
        return zscore(x.T, axis=0).T
    elif axis != 0:
        raise ValueError('axis must be either 0 or 1')

        z = x.copy()
        for c in z.columns():
            z[c] -= z[c].mean(axis=0)
            z[c] /= z[c].std(axis=0)
        return z
