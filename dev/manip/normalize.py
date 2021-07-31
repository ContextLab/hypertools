# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np


@dw.decorate.apply_stacked
def normalize(x, axis=0):
    if axis == 1:
        return normalize(x.T, axis=0).T
    elif axis != 0:
        raise ValueError('axis must be either 0 or 1')

        z = x.copy()
        for c in z.columns():
            z[c] -= z[c].min(axis=0)
            z[c] /= z[c].max(axis=0)
        return z
