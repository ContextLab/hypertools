# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

from .common import Manipulator


# noinspection PyShadowingBuiltins
@dw.decorate.apply_stacked
def fitter(data, axis=0):
    if axis == 1:
        return dw.core.update_dict(fitter(data.T, axis=0), {'transpose': True})
    elif axis != 0:
        raise ValueError('axis must be either 0 or 1')

    mean = pd.Series(index=data.columns)
    std = pd.Series(index=data.columns)

    for c in data.columns:
        mean[c] = data[c].mean(axis=0)
        std[c] = data[c].std(axis=0)

    return {'mean': mean, 'std': std, 'axis': axis, 'transpose': False}


# noinspection DuplicatedCode
@dw.decorate.apply_stacked
def transformer(data, **kwargs):
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])}))

    assert kwargs['axis'] == 0, ValueError('invalid transformation')

    z = data.copy()
    for c in z.columns:
        z[c] -= kwargs['mean'][c]
        z[c] /= kwargs['std'][c]
    return z


class ZScore(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0):
        required = ['transpose', 'mean', 'std', 'axis']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None,
                         required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
